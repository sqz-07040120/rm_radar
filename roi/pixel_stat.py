import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def mouseColor(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(img[y, x])

total_pixel = [0 for i in range(256)]
filepath = '../images/roi'
for root, dir, files in os.walk(filepath):
    root, dir, files = root, dir, files
index = 0
pict = 0
for imgpath in files:
    time1 = time.time()
    index = index + 1
    print(str(index) + '/' + str(len(files)))
    img_pixle = [0 for i in range(256)]
    img = cv2.imread(root + '/' + imgpath)
    src_img = img.copy()
    # 统计图像中像素点的像素值
    for i in range(256):
        img_pixle[i] = img[img == i].size
        total_pixel[i] = total_pixel[i] + img_pixle[i]
    time2 = time.time()
# 方法一 : 滑动均值最小，取整个区间的波谷
    # 计算阈值
    filtsize = 5
    pixsum = [0 for i in range(len(img_pixle)//filtsize)]
    for i in range(len(img_pixle)//filtsize):
        pixsum[i] = np.sum(img_pixle[i*filtsize:i*filtsize+filtsize])
    thresh = pixsum.index(min(pixsum)) * filtsize

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, bina_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    print('峰值为:', thresh)

# 方法2 : 最大类间方差法
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh2, bina_img2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print('峰值为:', thresh)  # thresh为OTSU算法得出的阈值

    print('tt对比' + str(thresh) + ' ' + str(thresh2))
    if abs((thresh - thresh2) / thresh ) > 0.5:
        print('较大差' + str(thresh) + ' ' + str(thresh2))

    print('fps: ' + str(1/(time2 - time1)) + ' time: ' + str(time2 - time1))
    if (index > pict) & (thresh > 220):
        cv2.namedWindow("src_img", 0)
        cv2.imshow('src_img', src_img)

        cv2.namedWindow("Color Picker", 0)
        cv2.setMouseCallback("Color Picker", mouseColor)
        cv2.imshow('Color Picker', img)

        cv2.namedWindow("bina img", 0)
        cv2.imshow('bina img', bina_img)
        cv2.namedWindow("bina img2", 0)
        cv2.imshow('bina img2', bina_img2)

        plt.bar([i for i in range(256)], img_pixle, width=1, edgecolor="white", linewidth=0.7)
        plt.xlabel('0-255')
        plt.ylabel('range')
        plt.title(r'pixel count')
        plt.show()

        cv2.waitKey()


fig, ax = plt.subplots()
plt.bar([i for i in range(256)], total_pixel, width=1, edgecolor="white", linewidth=0.7)
plt.xlabel('0-255')
plt.ylabel('range')
plt.title(r'pixel count')
plt.show()
