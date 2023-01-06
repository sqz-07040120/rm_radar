import cv2
import numpy as np
from utils.Util import print_camera_message,set_laptop_camera

import protobuf_serial.message_pb2 as mp

# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# print_camera_message(camera)
# set_laptop_camera(camera)
# 判断视频是否打开
# if (camera.isOpened()):
#     print('Open')
# else:
#     print('摄像头未打开')

camera = cv2.VideoCapture('../images/rm上交雷达左视角.mp4')
set_laptop_camera(camera)
# 测试用,查看视频size
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:' + repr(size))

# 高斯模糊核
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((5, 5), np.uint8)
background = None

while True:
    # 读取视频流
    grabbed, frame_lwpCV = camera.read()
    # 对帧进行预处理，先转灰度图，再进行高斯滤波。
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray_lwpCV
        continue
    # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    kernel = np.ones((5, 5), np.uint8)
    # closing = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    cv2.erode(diff, kernel, iterations=2)
    # diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀
    # 显示矩形框
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓

    # 传递飞坡车出现的信号
    if(len(contours)!=0):
        Flyslope = mp.Flyslope()
        Flyslope.FSalarm = "飞坡警告"
        f = open("飞坡警告", "wb")
        f.write(Flyslope.SerializeToString())
        f.close()


    for c in contours:
        if cv2.contourArea(c) < 200:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        # 设置感兴趣区，只对这部分内出现的移动物体做标记
        if x>(np.size(diff,1)/3) and y>(np.size(diff,0)/3) and (x+w)<(np.size(diff,1)/2-20) and (y+h)<(np.size(diff,0)/2+20):
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 10)
        # cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 操作后重新设置backgrund，使上一帧图像作为下一帧图像的背景
    background = gray_lwpCV

    cv2.imshow('contours', frame_lwpCV)
    cv2.imshow('diff', diff)
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord(' '):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()