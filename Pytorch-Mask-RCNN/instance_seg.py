# -*- coding: gbk -*-
import argparse
import json
import os
from glob import glob

import pandas as pd
import torch
import torchvision
import time
import numpy as np
from PIL import Image

from torchvision import transforms
from tqdm import tqdm

from detection.draw_box_utils import draw_objs

indices = {"1": "person", "2": "bicycle", "3": "car", "4": "motorcycle", "6": "bus", "8": "truck"}


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


@torch.inference_mode()
def predict(args):
    # Setup dataloader
    # image_files = []
    # if os.path.isdir(args.img_folder):
    #     for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
    #         files = glob(os.path.join(args.img_folder, '**/*.%s' % (ext)), recursive=True)
    #         if len(files) > 0:
    #             image_files.extend(files)
    # elif os.path.isfile(args.img_folder):
    #     image_files.append(args.img_folder)
    #
    # unique_imgs = list(set(image_files))

    pd_image = pd.read_csv(args.img_files)
    all_image_list = []
    for index, row in pd_image.iterrows():
        i_image_positive = eval(row['positive'])
        i_image_negative = eval(row['negative'])
        all_image_list.extend(i_image_positive)
        all_image_list.extend(i_image_negative)

    final_images = list(set(all_image_list))

    with open('../dict_img_path.json', 'r') as file:
        img_path_relation = json.load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    pd_semantic = pd.read_csv('../results/all_segmentation.csv')
    pd_semantic.insert(pd_semantic.shape[1], 'people_cnt', 0)
    pd_semantic.insert(pd_semantic.shape[1], 'vehicle_cnt', 0)

    for img_name in tqdm(final_images):
        if img_name in img_path_relation.keys():
            img_path = args.img_folder + img_path_relation[img_name]
            ori_img = Image.open(img_path).convert('RGB')
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(ori_img).to(device)

            with torch.no_grad():
                predictions = model([img])[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)

            num_people = sum(1 for cls, score in zip(predict_classes, predict_scores) if (cls == 1 and score >= args.box_conf))
            num_vehicles = sum(1 for cls, score in zip(predict_classes, predict_scores) if (cls in [2, 3, 4, 6, 8] and score >= args.box_conf))

            mask = (pd_semantic['id'] == img_name)
            pd_semantic.loc[mask, 'people_cnt'] = num_people
            pd_semantic.loc[mask, 'vehicle_cnt'] = num_vehicles

            # print("Image: {}\t".format(img_name))
            # print("预测人数： {}\t".format(num_people))
            # print("预测车辆数： {}".format(num_vehicles))

            if args.save_img:
                # 保存预测的图片结果
                output_img_path = os.path.join(args.out_folder, img_name)
                plot_img = draw_objs(ori_img,
                                     boxes=predict_boxes,
                                     classes=predict_classes,
                                     scores=predict_scores,
                                     masks=predict_mask,
                                     category_index=indices,
                                     box_thresh=args.box_conf,
                                     mask_thresh=args.mask_conf,
                                     line_thickness=1,
                                     font='arial.ttf',
                                     font_size=10)
                plot_img.save(output_img_path)

    pd_semantic.to_csv('../results/seg_instance.csv', sep=',', index=False)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="预测")
    parse.add_argument('--weights-path', type=str, default='./MaskRCNN_9.pth')
    parse.add_argument('--img-folder', type=str, default='../images/')
    parse.add_argument('--img-files', type=str, default='../pd_valid_sample_sets.csv')
    parse.add_argument('--out-folder', type=str, default='./out')
    parse.add_argument('--box-conf', type=float, default=0.7)
    parse.add_argument('--mask-conf', type=float, default=0.5)
    parse.add_argument('--save-img', type=float, default=False)

    args = parse.parse_args()
    predict(args)
