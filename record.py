import numpy as np
# from PIL import ImageGrab
import cv2
from utils import Util

left_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
Util.record_camera(left_camera)
Util.print_camera_message(left_camera)
width, high = left_camera.get(cv2.CAP_PROP_FRAME_WIDTH),left_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取屏幕的宽和高
fourcc = cv2.VideoWriter_fourcc(*'I420')  # 设置视频编码格式
fps = 60  # 设置帧率
video = cv2.VideoWriter('./record_image/test1.avi', fourcc, fps, (int(width), int(high)), True)
while True:
    _, frame = left_camera.read()
    video.write(frame)
    # cv2.imshow("aa",frame)
    key = cv2.waitKey(1) & 0xFF
    # 按' '健退出循环
    if key == ord(' '):
        break
video.release()  # 释放缓存，持久化视频
