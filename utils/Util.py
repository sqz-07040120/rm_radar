import cv2
import time
import numpy as np

def completesite_frame(img):
    cv2.rectangle(img, (45, 306), (73, 446), (0, 255, 0), 2)
    cv2.polylines(img, np.array([[[74, 285], [113, 284], [113, 339], [93, 356], [75, 356]]]), 1, (0, 255, 0), 2, 8, 0)


def cpsite_show(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)# 设置窗口大小
    # cv2.resize(img, (500, 1000), interpolation=cv2.INTER_CUBIC)
    cv2.resizeWindow(name, 500, 1000)
    cv2.moveWindow(name, 0, 0)  # 设置窗口在电脑屏幕中的位置
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #设置全屏显示
    cv2.imshow(name,img)

def detect_show(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)# 设置窗口大小
    cv2.resize(img, (1250, 760), interpolation=cv2.INTER_CUBIC)
    cv2.resizeWindow(name, 1250, 760)
    cv2.moveWindow(name, 490, 0)  # 设置窗口在电脑屏幕中的位置
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #设置全屏显示
    cv2.imshow(name,img)

def all_show(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)# 设置窗口大小
    cv2.resize(img, (2560, 1600), interpolation=cv2.INTER_CUBIC)
    cv2.resizeWindow(name, 2560, 1600)
    cv2.moveWindow(name, 0, 0)  # 设置窗口在电脑屏幕中的位置
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #设置全屏显示
    cv2.imshow(name,img)


def screen_show(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # 设置窗口大小
    cv2.resize(img, (1250, 240), interpolation=cv2.INTER_CUBIC)
    cv2.resizeWindow(name, 1250, 240)
    cv2.moveWindow(name, 462, 755)  # 设置窗口在电脑屏幕中的位置
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #设置全屏显示
    cv2.imshow(name, img)

def ccs(y,size_y):
    # 转换图像的坐标系 change_coordinate_system
    return size_y - y

def deter_point(pointrange,x,y,size_y):
    # 画面左上角为坐标原点, xup,yup,左上角点的坐标,xdown,ydown,右下角点的坐标
    (x1,y1,x2,y2,x3,y3,x4,y4)=pointrange
    # 转换图像的坐标系
    y1 = ccs(y1, size_y)
    y2 = ccs(y2, size_y)
    y3 = ccs(y3, size_y)
    y4 = ccs(y4, size_y)
    yx = ccs(y, size_y)
    #第一次筛选，直接对y
    up_y = y1 if y1 > y4 else y4
    down_y = y2 if y2 < y3 else y3
    if yx>up_y or yx<down_y:
        return False
    #第二次筛选，对x
    k1 = (x2 - x1)/(y2 - y1)
    b1 = k1*y1 - x1
    l1_x = yx*k1 - b1
    k2 = (x3 - x4)/(y3 - y4)
    b2 = k2*y4 - x4
    l2_x = yx*k2 - b2
    if x>l1_x and x<l2_x:
        return True
    else:
        return False


# def deter_point(pointrange,x,y,size_y):
#     (x1,y1,x2,y2,x3,y3,x4,y4)=pointrange
#     # 转换图像的坐标系
#     y1 = ccs(y1, size_y)
#     y2 = ccs(y2, size_y)
#     y3 = ccs(y3, size_y)
#     y4 = ccs(y4, size_y)
#     yx = ccs(y, size_y)
#     up_y = y3 if y3 < y4 else y4
#     down_y = y2 if y1 < y2 else y1
#     if yx>up_y or yx<down_y:
#         return False
#     k1 = (x4 - x1)/(y4 - y1)
#     b1 = x1 - k1*y1
#     l1_x = yx*k1 + b1
#     k2 = (x3 - x2)/(y3 - y2)
#     b2 = x2 - k2*y2
#     l2_x = yx*k2 + b2
#     if x>l1_x and x<l2_x:
#         return True
#     else:
#         return False

def diff(background,gray_lwpCV,es):
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    kernel = np.ones((5, 5), np.uint8)
    # closing = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    cv2.erode(diff, kernel, iterations=2)
    diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀
    return diff

def print_camera_message(cap):
    print('视频编码格式'+str(cap.get(6)))
    print('帧率'+str(cap.get(5)))
    print('宽高'+str(cap.get(3))+','+str(cap.get(4)))
    print('曝光'+str(cap.get(15)))
    print('增益'+str(cap.get(cv2.CAP_PROP_GAIN)))
    print('亮度' + str(cap.get(cv2.CAP_PROP_BRIGHTNESS)))
    print('白平衡'+str(cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)))

# def final_black_camera(cap):
#     ret = cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
#     # ret = cap.set(5, 90)
#     # time.sleep(0.5)
#     ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
#     ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1535)
#     # ret = cap.set(cv2.CAP_PROP_AUTO_WB, 1)
#
#     # cv2.CV_CAP_PROP_WHITE_BALANCE_BLUE_U
#
#     # ret = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.39)
#     ret = cap.set(cv2.CAP_PROP_EXPOSURE, -4)
#     # 亮度，   增益 ，    白平衡
#     ret = cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
#     ret = cap.set(cv2.CAP_PROP_GAIN, 30)
#     ret = cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)
#     # 视频编码格式 - 466162819.0
#     # 宽高3840, 2160
#     # 曝光 - 4.0
#     # 增益30
#     # 白平衡4600

def final_black_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1535)

    capture.set(cv2.CAP_PROP_FPS, 30)#帧率 帧/秒

    capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)#亮度
    capture.set(cv2.CAP_PROP_GAIN,100)#增益
    capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)#白平衡
    capture.set(cv2.CAP_PROP_CONTRAST,1)#对比度
    capture.set(cv2.CAP_PROP_SHARPNESS, 7)  #锐化（清晰度）

    capture.set(cv2.CAP_PROP_SATURATION, 60)#饱和度 50
    capture.set(cv2.CAP_PROP_GAMMA, 100)    #伽马校正
    capture.set(cv2.CAP_PROP_HUE, 0)#色调 50

    capture.set(cv2.CAP_PROP_EXPOSURE, -6)#曝光 50 获取摄像头参数

def record_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1535)

    capture.set(cv2.CAP_PROP_FPS, 30)#帧率 帧/秒

    capture.set(cv2.CAP_PROP_BRIGHTNESS, 0)#亮度
    # capture.set(cv2.CAP_PROP_GAIN,100)#增益
    capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)#白平衡
    capture.set(cv2.CAP_PROP_CONTRAST,3)#对比度
    capture.set(cv2.CAP_PROP_SHARPNESS, 2)  #锐化（清晰度）

    capture.set(cv2.CAP_PROP_SATURATION, 62)#饱和度 50
    capture.set(cv2.CAP_PROP_GAMMA, 120)    #伽马校正
    capture.set(cv2.CAP_PROP_HUE, 0)#色调 50

    capture.set(cv2.CAP_PROP_EXPOSURE, -6)#曝光 50 获取摄像头参数

def set_laptop_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    capture.set(cv2.CAP_PROP_FPS, 30)  # 帧率 帧/秒

    capture.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # 亮度
    # capture.set(cv2.CAP_PROP_GAIN,100)#增益
    capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)  # 白平衡
    capture.set(cv2.CAP_PROP_CONTRAST, 32)  # 对比度
    capture.set(cv2.CAP_PROP_SHARPNESS, 3)  # 锐化（清晰度）

    capture.set(cv2.CAP_PROP_SATURATION, 64)  # 饱和度 50
    capture.set(cv2.CAP_PROP_GAMMA, 120) #伽马校正
    capture.set(cv2.CAP_PROP_HUE, 0)  # 色调 50

    capture.set(cv2.CAP_PROP_EXPOSURE, -6)  # 曝光 50 获取摄像头参数

def smallbox_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1535)

    capture.set(cv2.CAP_PROP_FPS, 30)  # 帧率 帧/秒

    capture.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # 亮度
    capture.set(cv2.CAP_PROP_GAIN,100) #增益
    capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)  # 白平衡
    capture.set(cv2.CAP_PROP_CONTRAST, 1)  # 对比度
    capture.set(cv2.CAP_PROP_SHARPNESS,0)  # 锐化（清晰度）

    capture.set(cv2.CAP_PROP_SATURATION, 60)  # 饱和度 50
    capture.set(cv2.CAP_PROP_GAMMA, 100)  # 伽马校正
    capture.set(cv2.CAP_PROP_HUE, 0)  # 色调 50

    capture.set(cv2.CAP_PROP_EXPOSURE, -6)  # 曝光 50 获取摄像头参数

def bigbox_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1535)

    capture.set(cv2.CAP_PROP_FPS, 30)  # 帧率 帧/秒

    capture.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # 亮度
    # capture.set(cv2.CAP_PROP_GAIN,100)#增益
    capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)  # 白平衡
    capture.set(cv2.CAP_PROP_CONTRAST, 3)  # 对比度
    capture.set(cv2.CAP_PROP_SHARPNESS,9)  # 锐化（清晰度）

    capture.set(cv2.CAP_PROP_SATURATION, 62)  # 饱和度 50
    capture.set(cv2.CAP_PROP_GAMMA, 120)  # 伽马校正
    capture.set(cv2.CAP_PROP_HUE, 0)  # 色调 50

    capture.set(cv2.CAP_PROP_EXPOSURE, -6)  # 曝光 50 获取摄像头参数

def test_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1535)

    capture.set(cv2.CAP_PROP_FPS, 30)  # 帧率 帧/秒

    capture.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # 亮度
    # capture.set(cv2.CAP_PROP_GAIN,100)#增益
    capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4600)  # 白平衡
    capture.set(cv2.CAP_PROP_CONTRAST, 3)  # 对比度
    capture.set(cv2.CAP_PROP_SHARPNESS,9)  # 锐化（清晰度）

    capture.set(cv2.CAP_PROP_SATURATION, 62)  # 饱和度 50
    capture.set(cv2.CAP_PROP_GAMMA, 120)  # 伽马校正
    capture.set(cv2.CAP_PROP_HUE, 0)  # 色调 50

    capture.set(cv2.CAP_PROP_EXPOSURE, -6)  # 曝光 50 获取摄像头参数

