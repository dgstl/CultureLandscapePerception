import time

import cv2
import numpy as np
from PIL import Image
from scipy.stats import tstd
from skimage import color

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


def calculate_average_brightness(image_path):
    # 计算平均亮度
    # 打开图片
    with Image.open(image_path) as img:
        # 将图片转换为RGB模式，确保每个像素都能被访问
        img = img.convert('RGB')
        pixels = img.getdata()  # 获取像素数据

        # 初始化总和变量
        total_brightness = 0

        # 遍历所有像素
        for r, g, b in pixels:
            # 应用公式计算亮度
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            total_brightness += brightness

        # 计算平均亮度
        average_brightness = total_brightness / img.width / img.height

        return average_brightness


def calculate_color_perceptibility(image_path):
    # 加载图像并转换为浮点数，以便进行计算
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    # 将RGB色彩空间转换为CIELUV色彩空间
    image_LUV = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)

    # 分离L, U, V通道
    L, U, V = cv2.split(image_LUV)

    # 计算每个像素的饱和度：根据U和V通道计算色度
    chroma = np.sqrt(U ** 2 + V ** 2)

    # 计算饱和度的平均值和标准差
    mean_saturation = np.mean(chroma)
    std_saturation = tstd(chroma.flatten())

    # 计算色彩感知度
    color_perceptibility = mean_saturation + std_saturation

    return color_perceptibility


def find_road_width(segmentation):
    # 创建一个二值图像，只有道路部分为白色
    road_binary = (segmentation == 0).astype(np.uint8) * 255

    # 使用OpenCV找到轮廓
    contours, _ = cv2.findContours(road_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，返回None
    if not contours:
        return 0

    # 找到最大的轮廓（假设最大轮廓具有最多的像素）
    max_contour = max(contours, key=cv2.contourArea)

    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(max_contour)

    # 计算中心点
    center_x = x + w // 2
    center_y = y + h // 2

    # 计算中心点到轮廓两侧的最小距离
    left_distance = center_x - x
    right_distance = (x + w) - center_x

    # 计算中心点到轮廓上下的最小距离
    top_distance = center_y - y
    bottom_distance = (y + h) - center_y

    # 取这两个距离的最小值作为道路宽度
    road_width = min(left_distance + right_distance, top_distance + bottom_distance)

    return road_width


def calculate_average_building_height(segmentation):
    # 创建一个二值图像，只有建筑物部分为白色
    buildings_binary = (segmentation == 2).astype(np.uint8) * 255

    # 使用OpenCV找到轮廓
    contours, _ = cv2.findContours(buildings_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化一个列表来存储所有轮廓的高度
    heights = []

    # 遍历所有轮廓
    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 将高度（像素）添加到列表中
        if h > 50:
            heights.append(h)

    # 如果存在轮廓，则计算平均高度
    if heights:
        average_height = np.mean(heights)
    else:
        average_height = 0

    return average_height


def calculate_enclosure_degree(segmentation_image):
    """
        根据语义分割图像计算道路和建筑的像素宽度和高度。

        参数:
        segmentation_image (numpy.ndarray): 分割图像，其中每个像素的值代表其类别标签。
        road_label (int): 道路的标签。
        building_label (int): 建筑的标签。

        返回:
        dict: 包含道路和建筑的像素宽度和高度的字典。
        """
    # 查找道路和建筑物的像素坐标
    road_width = find_road_width(segmentation_image)
    building_height = calculate_average_building_height(segmentation_image)

    # 返回结果
    # print('road_pixel_width:', road_width, 'building_pixel_height:', building_height)
    if road_width > 0:
        return building_height / road_width
    else:
        return None


def cur_type_complex(lst):
    ci = 0.0
    if sum(lst) > 0:
        for ll in lst:
            ratio = ll / sum(lst)
            if ratio > 0:
                ci += ratio * np.log2(ratio)
    else:
        return 0
    return ci


def complexity(semanteme):
    # 0、公路，1、人行道，2、建筑物，3、墙壁，4、栅栏，5、杆子，6、信号灯，7、交通标识，8、植被，9、山坡，10、天空，11、人，12、骑行者，13、汽车，14、卡车，15、公交车，16、火车，17、摩托车，18、自行车
    types = {'natural': ['vegetation', 'terrain', 'sky'],
             'structural': ['buildings', 'wall', 'fence'],
             'dynamic': ['people', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bicycle'],
             'road': ['road', 'side way', 'pole', 'signal light', 'traffic sign']}

    seg_complex = 0.0
    for k, v in types.items():
        cur_ls = []
        for vv in v:
            cur_ls.append(semanteme[vv])
        seg_complex += cur_type_complex(cur_ls)

    return -seg_complex
