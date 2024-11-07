import json
import os

import numpy as np
import pandas as pd
from segmentation import seg, instance_segmentation
from index_calcu import calculate_average_brightness, calculate_color_perceptibility, calculate_enclosure_degree, complexity

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 5)


def cal_index(id, region, dimension, attr, sementics, img_path):
    if sementics['road'] + sementics['side way'] > 0:
        # vi = [(sementics['people'] + sementics['rider'] + sementics['people'] + sementics['car'] + sementics['truck'] + sementics['bus'] +
        #       sementics['train'] + sementics['motor'] + sementics['bicycle']) / (sementics['road'] + sementics['side way'])] * (sementics['people_cnt'] + sementics['vehicle_cnt'])
        vi = sementics['people_cnt'] + sementics['vehicle_cnt']
    else:
        vi = 0

    # 加载语义分割结果
    pred = np.load('images_np/' + id + '.npy')
    ei = calculate_enclosure_degree(pred)

    new = pd.DataFrame({'id': id,
                        'city': region,
                        'dimension': dimension,
                        'attr': attr,
                        'GI': sementics['vegetation'],
                        'BI': sementics['sky'],
                        'WI': sementics['side way'],
                        'EI': ei,
                        'VI': vi,
                        'Brightness': calculate_average_brightness(img_path),
                        'Colorfulness': calculate_color_perceptibility(img_path),
                        'Complexity': complexity(sementics)
                        },
                       index=[1])
    return new


def physical_visual():
    pd_segs = pd.read_csv('./results/seg_instance.csv', sep=',')
    pd_results = pd.DataFrame(columns=['id', 'city', 'dimension', 'attr', 'GI', 'BI', 'WI', 'EI', 'VI', 'Brightness', 'Colorfulness', 'Complexity'])
    with open('dict_img_path.json', 'r') as file:
        img_path_relation = json.load(file)

    pd_sample_sets = pd.read_csv('pd_valid_sample_sets.csv')
    for i, row in pd_sample_sets.iterrows():
        i_positives = eval(row['positive'])
        i_negatives = eval(row['negative'])
        for id in i_positives:
            if id in img_path_relation.keys():
                print(id)
                img_path = 'images/' + img_path_relation[id]
                sementics = pd_segs.loc[pd_segs['id'] == id].squeeze().to_dict()
                id_series = cal_index(id, row['region'], row['dimension'], 1, sementics, img_path)
                pd_results = pd_results.append(id_series, ignore_index=True)

        for id in i_negatives:
            if id in img_path_relation.keys():
                print(id)
                img_path = 'images/' + img_path_relation[id]
                sementics = pd_segs.loc[pd_segs['id'] == id].squeeze().to_dict()
                id_series = cal_index(id, row['region'], row['dimension'], 0, sementics, img_path)
                pd_results = pd_results.append(id_series, ignore_index=True)

    pd_results.to_csv('results/all_vectors.csv')


if __name__ == "__main__":
    physical_visual()
