# import cv2
# import numpy as np
#
#
# def segment_finger_nails(image_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#     # 转换为灰度图
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 使用阈值分割出指甲
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     # 形态学操作 - 先膨胀后腐蚀
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.dilate(thresh, kernel, iterations=1)
#     mask = cv2.erode(mask, kernel, iterations=1)
#     # 将掩码应用到原图
#     result = cv2.bitwise_and(image, image, mask=mask)
#     return result
#
#
# # 使用函数分割图像中的指甲
# image_path = 'C:/Users/lby/Desktop/dcsql/zj1.jpg'  # 替换为你的图片路径
# segmented_image = segment_finger_nails(image_path)
#
# # 显示结果
# cv2.imshow('Segmented Finger Nails', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
#
#
# def segment_nails(image_path):
#     # 读取图像
#     img = cv2.imread(image_path)
#     # 转换为灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 使用高斯滤波去除噪声
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # 应用阈值操作进行分割
#     _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
#     # 形态学操作 - 开运算以去除小噪声
#     kernel = np.ones((5, 5), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#     # 寻找轮廓
#     contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 为每个轮廓创建一个掩膜并进行绘制
#     for contour in contours:
#         # 计算轮廓的边界框
#         x, y, w, h = cv2.boundingRect(contour)
#         # 根据边界框绘制矩形轮廓
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     # 展示图像
#     cv2.imshow('Segmented Nails', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # 使用函数分割指甲
# segment_nails('C:/Users/lby/Desktop/dcsql/zj2.jpg')

import cv2
import numpy as np


def split_nail_area(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用阈值分割出指甲区域
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # 形态学操作 - 先膨胀后腐蚀
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=2)
    erosion = cv2.erode(dilation, kernel, iterations=2)
    # 找到轮廓
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓并绘制最大的轮廓
    max_contour_idx = 0
    max_area = 0
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour_idx = idx
    # 绘制轮廓
    cv2.drawContours(image, contours, max_contour_idx, (0, 255, 0), 3)
    # 显示图片
    cv2.imshow('Nail Area', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用函数分割指甲区域
# split_nail_area('img/zj1.jpg')
split_nail_area('img/zj2.jpg')







