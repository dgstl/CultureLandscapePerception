import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base_dir = '/Users/dgstl/Documents/Research/NJUPT/2023/Projects/RoutePlaning/datasource/place-pulse-2.0/'
region_city = {
    'asia_city': ['Taipei', 'Hong Kong', 'Tokyo', 'Kyoto', 'Bangkok'],
    'europe_city': ['Amsterdam', 'Paris', 'Berlin', 'Prague', 'Warsaw']
}


def plot_bin(scores, region, dim):
    # 计算直方图
    counts, bins = np.histogram(scores, bins='auto')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # 绘制频率分布图
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, counts, align='center', width=0.8 * (bins[1] - bins[0]), color='skyblue')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of ' + region + ' ' + dim)
    plt.xticks(bins)
    plt.show()


def get_samples(region, dimension, is_draw=False):
    all_dimension = pd.read_csv(base_dir + 'studies.txt', sep='\t')
    cities = pd.read_csv(base_dir + 'places.tsv', sep='\t')
    qscores = pd.read_csv(base_dir + '/qscores.tsv', sep='\t')

    # 美丽维度
    dim_id = all_dimension[all_dimension['study_question'] == dimension]['_id'].values[0]
    dim_socres = qscores[qscores['study_id'] == dim_id]['trueskill.score']

    # 特定区域城市和特定维度的得分数据
    region_city_id = [cities[cities['place_name'] == id]['_id'].values[0] for id in region_city[region]]
    print(region, region_city_id)
    region_city_dim = qscores[(qscores['study_id'] == dim_id) & (qscores['place_id'].isin(region_city_id))]

    print(dimension, '维度的图片数量：', len(dim_socres))
    print('该维度', region, '城市的图片数量：', len(region_city_dim))

    region_socres = region_city_dim['trueskill.score'].values
    if is_draw:
        plot_bin(region_socres, region, dimension)

    mix_score = np.mean(region_socres) - np.std(region_socres)
    max_score = np.mean(region_socres) + np.std(region_socres)
    print('取图片的对象临界点', mix_score, max_score)

    selected_region_dim_T = region_city_dim[region_city_dim['trueskill.score'] > max_score]
    selected_region_dim_F = region_city_dim[region_city_dim['trueskill.score'] < mix_score]

    positive_sample = selected_region_dim_T['location_id'].values
    negative_sample = selected_region_dim_F['location_id'].values
    print('正样本数量', len(positive_sample))
    print('负样本数量', len(negative_sample))

    # return positive_sample, negative_sample

    ps_votes_more = selected_region_dim_T[selected_region_dim_T['num_votes'] > 2]
    ns_votes_more = selected_region_dim_F[selected_region_dim_F['num_votes'] > 2]

    ps_votes_more_sorted = ps_votes_more.sort_values(by='trueskill.score', ascending=False)
    ns_votes_more_sorted = ns_votes_more.sort_values(by='trueskill.score', ascending=False)
    print('对比次数大于2的正样本数量', len(ps_votes_more_sorted))
    print('对比次数大于2的负样本数量', len(ns_votes_more_sorted))

    ps_votes_more_sorted['attr'] = 1
    ns_votes_more_sorted['attr'] = 0

    # 合并两个DataFrame并重新索引
    sample_all_connect = pd.concat([ps_votes_more_sorted, ns_votes_more_sorted], ignore_index=True)
    sample_all_connect['dim'] = dimension
    sample_all_connect['region'] = region
    cities = cities.rename(columns={'_id': 'place_id'})
    merged_df = pd.merge(sample_all_connect, cities, on='place_id', how='left')

    return (ps_votes_more_sorted['location_id'].values, ns_votes_more_sorted['location_id'].values,
            merged_df[['region', 'dim', 'attr', 'location_id', 'trueskill.score', 'place_name']])


# get_samples('asia_city', 'more beautiful')
def region_img_filter():
    """
    筛选出亚洲和欧洲各维度的正负样本集
    :return:
    """
    need_regions = ['asia_city', 'europe_city']
    need_dimensions = ['safer', 'more beautiful', 'more depressing', 'wealthier', 'more boring', 'livelier']
    region_dim_images = {}  #
    pd_region_dim_image = pd.DataFrame()

    for nr in need_regions:
        region_dim_images[nr] = {}
        for dim in need_dimensions:
            p, n, sample_d = get_samples(nr, dim)
            if dim not in region_dim_images[nr].keys():
                region_dim_images[nr][dim] = {}
            region_dim_images[nr][dim]['positive'] = p.tolist()
            region_dim_images[nr][dim]['negative'] = n.tolist()

            # new_row = pd.Series({'region': nr, 'dimension': dim, 'positive': p.tolist(), 'negative': n.tolist()})
            # pd_region_dim_image.loc[len(pd_region_dim_image)] = new_row
            pd_region_dim_image = pd.concat([pd_region_dim_image, sample_d], ignore_index=True)

    with open('dict_valid_sample_sets.json', 'w') as file:
        json.dump(region_dim_images, file)

    pd_region_dim_image.to_csv('pd_valid_sample_sets_new.csv', sep=',')


def gen_name_pics_relate():
    """
    生成图片id与图片全名的关联
    :return:
    """
    images_path = '/Users/dgstl/Documents/Research/NJUPT/2023/Projects/RoutePlaning/datasource/place-pulse-2.0/images'
    dict_img_path = {}
    for f in os.listdir(images_path):
        img_id = os.path.splitext(f)[0].split('_')[2]
        dict_img_path[img_id] = f
    with open('dict_img_path.json', 'w') as file:
        json.dump(dict_img_path, file)


region_img_filter()
# get_samples('asia_city', 'more beautiful', False)
