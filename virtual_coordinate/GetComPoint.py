import cv2
import numpy
from utils.Util import deter_point,print_camera_message
from utils import param,Util
import time
from PIL import Image
from utils.opencvmouse import check_location, count_zoom
from distent import coordinate,base_objpoints,leftedge_to_Rbase,leftedge_to_Radar,Bbase_to_Rbase,cameraMatrix,distCoeffs
from nets.yolo import YoloBody
import torch
from eval1 import yolo_eval, get_class

# -------------------------提醒---------------------------------
print("检查是否为相机模式")
print("检查是否开启录制")
# ------------------------选择数据是视频还是相机--------------------
mode = "video_test"
# mode = "camera"
# comside = 'red'
comside = 'blue'
# -----------------------------相机----------------------------------------------------
if mode == "camera":
                     # cv::CAP_FFMPEG或cv::CAP_IMAGES或cv::CAP_DSHOW
    forward_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    Util.final_black_camera(forward_camera)
    # Util.record_camera(forward_camera)
    print_camera_message(forward_camera)

    size_y = forward_camera.get(4)  # 后续图像处理使用

    # 判断视频是否打开
    if (forward_camera.isOpened()):
        print('Open')
    else:
        print('摄像头未打开')

# ------------------------------------视频-----------------------------------------------
if mode == "video_test":
    # 正视角飞坡和能量机关检测
    # forward_camera = cv2.VideoCapture("images/rm上交雷达左视角.mp4")
    # forward_camera = cv2.VideoCapture("images/2021-0801-2022-小组赛-四川大学VS南华大学-RD0.mkv")
    # forward_camera = cv2.VideoCapture("record_image/test.avi")
    forward_camera = cv2.VideoCapture("../record_image/广工test.mp4")

    size_y = forward_camera.get(4)#后续图像处理使用
    # 测试用,查看视频size
    size = (int(forward_camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(forward_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('size:' + repr(size))
# --------------------------------------------------------------------------------------
# ------------------------------场地缩略图------------------------------------------------
completesite = cv2.imread("../images/competesite.png")
completesite = cv2.resize(completesite, (500, 1000), interpolation=cv2.INTER_CUBIC) #蓝方
if comside == 'red':
    completesite = cv2.flip(completesite,-1) #红方
Util.completesite_frame(completesite)
completesite_origin = completesite.copy()

# ---------------------------------------------------标定------------------------------------------------
# 设置需要标的点的个数
# 3. 标定定位区域
# 全局变量
g_window_name = "Calibrate"  # 窗口名
g_window_wh = [1250, 760]  # 窗口宽高
g_location_win = [0, 0]  # 相对于大图，窗口在图片中的位置
location_win = [0, 0]  # 鼠标左键点击时，暂存g_location_win
g_location_click, g_location_release = [0, 0], [0, 0]  # 相对于窗口，鼠标左键点击和释放的位置
g_zoom, g_step = 1, 0.1  # 图片缩放比例和缩放系数
point_reverscircle_list=[]
grabbed, g_image_original = forward_camera.read(0) # 从相机截取一张图片用作标定
g_image_original = cv2.resize(g_image_original, (1250, 760), interpolation=cv2.INTER_CUBIC)
g_image_zoom = g_image_original.copy()  # 缩放后的图片
g_location_win = [0, 0]  # 相对于大图，窗口在图片中的位置
g_image_show = g_image_original[g_location_win[1]:g_location_win[1] + g_window_wh[1], g_location_win[0]:g_location_win[0] + g_window_wh[0]]  # 实际显示的图片
# 设置窗口
cv2.namedWindow(g_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(g_window_name, g_window_wh[0], g_window_wh[1])
cv2.moveWindow(g_window_name, 300, 180)  # 设置窗口在电脑屏幕中的位置
# 鼠标事件的回调函数
# OpenCV鼠标事件
def mouse(event, x, y, flags, param):
    global g_location_click, g_location_release, g_image_show, g_image_zoom, g_location_win, location_win, g_zoom
    if event == cv2.EVENT_LBUTTONDOWN:#左键点击
        x_o = int((x+g_location_win[0])/g_zoom)
        y_o = int((y+g_location_win[1])/g_zoom)
        xy = "%d,%d" % (x_o, y_o)
        cv2.circle(g_image_show, (x, y), 1, (255, 0, 0), thickness=2)
        cv2.putText(g_image_show, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), thickness = 1)
        point_reverscircle_list.append([x_o, y_o]) # 收集标定点
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
        g_location_click = [x, y]  # 右键点击时，鼠标相对于窗口的坐标
        location_win = [g_location_win[0], g_location_win[1]]  # 窗口相对于图片的坐标，不能写成location_win = g_location_win
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):  # 按住右键拖曳
        g_location_release = [x, y]  # 右键拖曳时，鼠标相对于窗口的坐标
        h1, w1 = g_image_zoom.shape[0:2]  # 缩放图片的宽高
        w2, h2 = g_window_wh  # 窗口的宽高
        show_wh = [0, 0]  # 实际显示图片的宽高
        if w1 < w2 and h1 < h2:  # 图片的宽高小于窗口宽高，无法移动
            show_wh = [w1, h1]
            g_location_win = [0, 0]
        elif w1 >= w2 and h1 < h2:  # 图片的宽度大于窗口的宽度，可左右移动
            show_wh = [w2, h1]
            g_location_win[0] = location_win[0] + g_location_click[0] - g_location_release[0]
        elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
            show_wh = [w1, h2]
            g_location_win[1] = location_win[1] + g_location_click[1] - g_location_release[1]
        else:  # 图片的宽高大于窗口宽高，可左右上下移动
            show_wh = [w2, h2]
            g_location_win[0] = location_win[0] + g_location_click[0] - g_location_release[0]
            g_location_win[1] = location_win[1] + g_location_click[1] - g_location_release[1]
        check_location([w1, h1], [w2, h2], g_location_win)  # 矫正窗口在图片中的位置
        g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] + show_wh[1], g_location_win[0]:g_location_win[0] + show_wh[0]]  # 实际显示的图片
    elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
        z = g_zoom  # 缩放前的缩放倍数，用于计算缩放后图片在窗口中的位置
        g_zoom = count_zoom(flags, g_step, g_zoom)  # 计算缩放倍数
        w1, h1 = [int(g_image_original.shape[1] * g_zoom), int(g_image_original.shape[0] * g_zoom)]  # 缩放图片的宽高
        w2, h2 = g_window_wh  # 窗口的宽高
        g_image_zoom = cv2.resize(g_image_o9riginal, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
        show_wh = [0, 0]  # 实际显示图片的宽高
        if w1 < w2 and h1 < h2:  # 缩放后，图片宽高小于窗口宽高
            show_wh = [w1, h1]
            cv2.resizeWindow(g_window_name, w1, h1)
        elif w1 >= w2 and h1 < h2:  # 缩放后，图片高度小于窗口高度
            show_wh = [w2, h1]
            cv2.resizeWindow(g_window_name, w2, h1)
        elif w1 < w2 and h1 >= h2:  # 缩放后，图片宽度小于窗口宽度
            show_wh = [w1, h2]
            cv2.resizeWindow(g_window_name, w1, h2)
        else:  # 缩放后，图片宽高大于窗口宽高
            show_wh = [w2, h2]
            cv2.resizeWindow(g_window_name, w2, h2)
        g_location_win = [int((g_location_win[0] + x) * g_zoom / z - x), int((g_location_win[1] + y) * g_zoom / z - y)]  # 缩放后，窗口在图片的位置
        check_location([w1, h1], [w2, h2], g_location_win)  # 矫正窗口在图片中的位置
        # print(g_location_win, show_wh)
        g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] + show_wh[1], g_location_win[0]:g_location_win[0] + show_wh[0]]  # 实际的显示图片
    cv2.imshow(g_window_name, g_image_show)
cv2.setMouseCallback(g_window_name, mouse)
cv2.waitKey()  # 不可缺少，用于刷新图片，等待鼠标操作
cv2.destroyAllWindows()
while True:
    # 读取视频流
    grabbed, frame_lwpCV = forward_camera.read()
    if grabbed == False:
        continue
    # -----------------------------------虚拟坐标--------------------------------------------------
    Lbase_points = point_reverscircle_list[:4]
    Rbase_points = point_reverscircle_list[4:8]
    Lbase_T = coordinate(base_objpoints, Lbase_points)
    Rbase_T = coordinate(base_objpoints, Rbase_points)
    Radia_Lbase = [Lbase_T[0][0], Lbase_T[1][0]]
    Radia_Rbase = [Rbase_T[0][0], Rbase_T[1][0]]
    O_base_R = (Lbase_T[0] + k2 * (Rbase_T[0] - Lbase_T[0]), Lbase_T[1])
    O_base_L = (Rbase_T[0] + k2 * (Rbase_T[0] - Lbase_T[0]), Rbase_T[1])
    # O__是O‘
    O__base_R = (O_base_R[0], Lbase_T[1])
    O__base_L = (O_base_L[0], Rbase_T[1])
    outpost_pic_L = (110, 403)
    outpost_pic_R = (387, 594)
    Ksc = outpost_pic_R[1] / Rbase_T[1]

    # 预测后得到的车的位置转换到场地缩略图之后的位置
    # pridict_x = coordinate()
    # pridict_y = coordinate()
    # cv2.circle(completesite, (int(pridict_x), int(pridict_y)), 1, (255, 0, 0), thickness=-1)
    # cv2.putText(completesite, '1'# vehicle_class
    #             , (int(pridict_x), int(pridict_y)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)

    cv2.circle(completesite, (int(Ksc * Radia_Lbase[0]), int(Ksc * Radia_Lbase[1])), 1, (0, 255, 0), thickness=4)
    cv2.putText(completesite, 'left'  # vehicle_class
                , (int(Ksc * Radia_Lbase[0]), int(Ksc * Radia_Lbase[1])), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0),
                thickness=2)
    cv2.circle(completesite, (int(Ksc * Radia_Rbase[0]), int(Ksc * Radia_Rbase[1])), 1, (0, 255, 0), thickness=4)
    cv2.putText(completesite, 'right'  # vehicle_class
                , (int(Ksc * Radia_Rbase[0]), int(Ksc * Radia_Rbase[1])), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0),
                thickness=2)
    cortime2 = time.time()
    # -------------------------------------------------------------------------------------------
    Util.cpsite_show('completesite', completesite)
    Util.detect_show('contours', frame_lwpCV)
    key = cv2.waitKey(1) & 0xFF
    # 按' '健退出循环
    if key == ord(' '):
        break