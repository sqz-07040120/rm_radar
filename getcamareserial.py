import cv2
from utils import Util

cameras = []
imgs = []
for i in range(10):
    cameras.append(cv2.VideoCapture(i, cv2.CAP_DSHOW))
    bools, img = cameras[i].read(0)
    if bools == True:
        imgs.append(img)
        cv2.imshow(str(i), imgs[i])
    else:
        imgs.append(1)
cv2.waitKey()

while True:
    for i,c in enumerate(cameras):

        # 读取视频流
        grabbed, img = c.read()
        if grabbed == False:
            print('False')
            # continue
            break

        name = str(i)
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)# 设置窗口大小
        # cv2.resize(img, (2560, 1600), interpolation=cv2.INTER_CUBIC)
        # cv2.resizeWindow(name, 2560, 1600)
        # cv2.moveWindow(name, 0, 0)  # 设置窗口在电脑屏幕中的位置
        cv2.imshow(name,img)

    key = cv2.waitKey(1) & 0xFF
    # 按' '健退出循环
    if key == ord(' '):
        break

for c in cameras:
    c.release()
    print('released')
