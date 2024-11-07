import json

import pandas as pd
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

label_idx = {
    'city': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # 0、公路，1、人行道，2、建筑物，3、墙壁，4、栅栏，5、杆子，6、信号灯，7、交通标识，8、植被，9、山坡，10、天空，11、人，12、骑行者，13、汽车，14、卡车，15、公交车，16、火车，17、摩托车，18、自行车
    'ade20k': [4, 2, 1, 6, 11, 33, 12, 80]
}


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default='test_results',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default='checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def cal_index(predict):
    # 上面矩阵中每一个元素就代表一个像素点，元素的值代表像素的分类，
    # 有了这个信息就可以计算绿视率与天空率了。在cityscapes训练集中，vegetation 的label是8，sky的label是10。因此，我们通过以下代码便可计算出绿视率与天空率。
    labels = label_idx['city']
    size = predict.shape[0] * predict.shape[1]
    indicators = []
    for lab in labels:
        val = (predict == lab)
        ratio = len(predict[val]) / size
        indicators.append(ratio)

    # print('绿视率, 天空率, 建筑物占比, 道路占比, 人行道占比, fence, 人, 车辆分别为:', str(indicators))
    return indicators


def main(input, output):
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    # image_files = []
    # if os.path.isdir(input):
    #     for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
    #         files = glob(os.path.join(input, '**/*.%s' % (ext)), recursive=True)
    #         if len(files) > 0:
    #             image_files.extend(files)
    # elif os.path.isfile(input):
    #     image_files.append(input)
    pd_image = pd.read_csv(input)
    all_image_list = []
    for index, row in pd_image.iterrows():
        i_image_positive = eval(row['positive'])
        i_image_negative = eval(row['negative'])
        all_image_list.extend(i_image_positive)
        all_image_list.extend(i_image_negative)

    final_images = list(set(all_image_list))
    print(len(all_image_list), len(final_images))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    all_seg_ratio = pd.DataFrame(columns=['id', 'road', 'side way', 'buildings', 'wall', 'fence', 'pole', 'signal light', 'traffic sign',
                                          'vegetation', 'terrain', 'sky', 'people', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bicycle'])

    with torch.no_grad():
        model = model.eval()
        # for img_path in tqdm(image_files):
        #     ext = os.path.basename(img_path).split('.')[-1]
        #     img_name = os.path.basename(img_path)[:-len(ext) - 1]
        base_path = '../images/'
        with open('../dict_img_path.json', 'r') as file:
            img_path_relation = json.load(file)

        for img_name in tqdm(final_images):
            if img_name in img_path_relation.keys():
                img_path = base_path+img_path_relation[img_name]
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0)  # To tensor of NCHW
                img = img.to(device)

                pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
                # 将语义分割结果保存为 .npy 文件
                np.save(os.path.join(output, img_name+'.npy'), pred)

                # 计算每个语义的占比
                ratio = cal_index(pred)
                ratio.insert(0, img_name)
                # 追加数据并重新赋值
                all_seg_ratio = all_seg_ratio.append(pd.Series(ratio, index=all_seg_ratio.columns), ignore_index=True)

                # colorized_preds = decode_fn(pred).astype('uint8')
                # colorized_preds = Image.fromarray(colorized_preds)
                # if opts.save_val_results_to:
                #     colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))

    all_seg_ratio.to_csv('../results/all_segmentation.csv', sep=',', index=False)


if __name__ == '__main__':
    main('../pd_valid_sample_sets.csv', 'D:\\Project\\CityPerception\\images_np')
