#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
# import numpy as np
import time
# from PIL import Image, ImageDraw, ImageFont

# def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
#     if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype("font/simsun.ttc",
#                                    textSize,
#                                    encoding="utf-8")
#     # 绘制文本
#     draw.text((left, top), text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

cam = cv2.VideoCapture(0)  # 读取摄像头
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
scale = 0

# 背景减除
fg = cv2.createBackgroundSubtractorMOG2()  # 实例化高斯混合模型

while True:
    time.sleep(0.02)
    ret, img = cam.read()
    if not ret:
        break

    # canny 边缘检测
    image = img.copy()
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯滤波
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)  # 灰度图
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)  # x方向梯度
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)  # y方向梯度
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    cv2.imshow("Canny", edge_output)

    # 背景减除
    fgmask = fg.apply(edge_output)  # 使用高斯混合模型
    cv2.imshow("fgmask", fgmask)

    # 闭运算
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4),
                                      (-1, -1))  # 定义结构元素，卷积核
    vline = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1), (-1, -1))
    result = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, hline)  # 水平方向
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, vline)  # 垂直方向
    cv2.imshow("result", result)

    # erodeim = cv2.erode(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations=1)  # 腐蚀
    dilateim = cv2.dilate(result,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
                          iterations=1)  # 膨胀
    # 查找轮廓
    contours, hier = cv2.findContours(dilateim, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.contourArea(c) > 1200:
            (x, y, w, h) = cv2.boundingRect(c)
            if scale == 0:
                scale = -1
                break
            scale = w / h
            cv2.putText(image, "scale:{:.3f}".format(scale), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (255, 0, 0), 1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            image = cv2.fillPoly(image, [c], (255, 255, 255))  # 填充

    # 根据人体比例判断
    if scale > 0 and scale < 1:
        # img = cv2ImgAddText(img, "Walking 行走中", 10, 20, (255, 0, 0), 30)  # 行走中
        cv2.putText(img, "Walking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)  # 行走中
    # if scale > 0.9 and scale < 2:
    #     img = cv2ImgAddText(img, "Falling 中间过程", 10, 20, (255, 0, 0),
    #                         30)  # 跌倒中
    #     # cv2.putText(img, "Falling 跌倒中", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#跌倒中
    if scale > 2:
        # img = cv2ImgAddText(img, "Falled 跌倒了", 10, 20, (255, 0, 0), 30)  # 跌倒了
        cv2.putText(img, "Falled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)  # 跌倒了

    cv2.imshow('test', image)
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cam.release()
cv2.destroyAllWindows()
