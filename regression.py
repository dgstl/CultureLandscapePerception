import numpy as np
import statsmodels.api as sm
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2

# 假设df是一个pandas DataFrame，其中包含你的数据
# X是自变量（特征），y是二元因变量
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 10)


def model(Region_Dim_data, stragety):
    Region_Dim_data['GI'] = Region_Dim_data['terrain'] + Region_Dim_data['GI']
    if stragety == 'vector':
        region_x = Region_Dim_data[['GI', 'BI', 'WI', 'EI', 'VI', 'Brightness', 'Colorfulness', 'Complexity']]  # WI
    elif stragety == 'semanteme':
        region_x = Region_Dim_data[['road', 'side way', 'buildings', 'wall', 'fence', 'pole', 'signal light', 'traffic sign',
                                    'vegetation', 'terrain', 'sky', 'people', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bicycle']]  # 示例自变量

    region_y = Region_Dim_data['attr']  # 二元因变量

    # 添加常数项以拟合截距
    X_with_const = sm.add_constant(region_x)

    # 构建模型并拟合数据
    model = sm.Logit(region_y, X_with_const).fit()

    # 计算模型预测的概率值
    pred_prob = model.predict(X_with_const)

    # 计算偏差残差，即 y - pred_prob
    # 对于逻辑回归，偏差残差是实际的y值（0或1）减去预测的概率
    residuals = region_y - pred_prob

    # 选择一个自变量绘制散点图，这里假设我们关注第一个自变量
    x_variable = region_x['GI']
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(6, 5))

    # 绘制散点图
    plt.scatter(x_variable, residuals, alpha=0.5, c='green')
    plt.title('Scatter plot of residuals vs. GI independent variable')
    plt.xlabel('Independent Variable')
    plt.ylabel('Deviance Residuals')

    plt.savefig('scatter-GI.png')

    # # 执行H-L检验
    # hl_stat, p_value = hosmer_lemeshow_test(model, region_x, region_y)
    # print(f"Hosmer-Lemeshow test statistic: {hl_stat}, p-value: {p_value}")
    #
    # 输出回归结果摘要
    print(model.summary())

    # # 计算优势比(OR)
    # # params = model.params
    # # conf = model.conf_int()
    # # conf['OR'] = params
    # # conf.columns = ['2.5%', '97.5%', 'OR']
    # # odds_ratio = np.exp(conf)
    # # print(odds_ratio)
    #
    # # 似然比检验的卡方值和p值
    # llr_chi2 = model.llr
    # llr_pvalue = model.llr_pvalue
    #
    # # 自由度
    # df_model = model.df_model
    #
    # # AIC和BIC值
    # aic = model.aic
    # bic = model.bic
    #
    # # 打印结果
    # print(f"似然比卡方值: {llr_chi2}")
    # print(f"自由度: {df_model}")
    # print(f"p值: {llr_pvalue}")
    # print(f"AIC值: {aic}")
    # print(f"BIC值: {bic}")


def hosmer_lemeshow_test(model, X, y, groups=10):
    """
    进行Hosmer-Lemeshow检验。

    参数:
    - model: 已拟合的statsmodels逻辑回归模型。
    - X: 自变量，应为带截距的DataFrame或numpy数组。
    - y: 因变量，应为Series或数组。
    - groups: 分组的数量，默认为10。

    返回:
    - H-L检验的统计量和p值。
    """
    # 预测概率
    probas = model.predict(X)

    # 计算分位数以分组
    deciles = pd.qcut(probas, groups, duplicates='drop')

    # 分组后的数据框
    data = pd.DataFrame({'probas': probas, 'y': y})
    data['deciles'] = pd.qcut(data['probas'], q=groups, duplicates='drop')

    # 计算每组的观测值和预期值
    grouped = data.groupby('deciles')
    obs = grouped['y'].sum()
    total = grouped['y'].count()
    exp = grouped['probas'].mean() * total

    # 计算Hosmer-Lemeshow统计量
    hl_stat = ((obs - exp) ** 2 / (exp * (1 - grouped['probas'].mean()))).sum()

    # 计算自由度
    df = groups - 2

    # 计算p值
    p_value = 1 - chi2.cdf(hl_stat, df)

    return hl_stat, p_value


def data_8_variable(dimension):
    # df = pd.read_csv('results/all_vectors.csv', sep=',')
    df_vectors = pd.read_csv('results/all_vectors-good.csv', sep=',')
    df_segs = pd.read_csv('results/all_segmentation.csv', sep=',')
    df = pd.merge(df_vectors, df_segs, on='id', how='inner')  # `how`参数确定合并方式，这里使用'inner'代表内连接
    # df = df_vectors[df_vectors['EI'].notnull()]
    df = df[(df['EI'] < 6.0) & df['VI'].notnull()]

    Asia = df[df['city'] == 'asia_city']
    Europe = df[df['city'] == 'europe_city']

    Asia_dim = Asia[Asia['dimension'] == dimension]
    Europe_dim = Europe[Europe['dimension'] == dimension]

    return Asia_dim, Europe_dim


dimen = 'more beautiful'  # 'safer', 'more beautiful', 'more depressing', 'wealthier', 'more boring', 'livelier'
Asia_data, Europe_data = data_8_variable(dimen)
print(dimen, Asia_data.shape, Europe_data.shape)
model(Europe_data, 'vector')