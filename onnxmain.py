import cv2
import numpy as np
from utils.Util import deter_point,print_camera_message
from utils import param,Util
import time
from PIL import Image
from utils.opencvmouse import check_location, count_zoom
from virtual_coordinate.distent import coordinate,base_objpoints,leftedge_to_Rbase,leftedge_to_Radar,Bbase_to_Rbase
from nets.yolo import YoloBody
import torch
import onnx
import onnxruntime
from eval1 import yolo_eval, get_class

import pycuda.autoinit

# -------------------------提醒---------------------------------
print("检查是否为相机模式")
print("检查是否开启录制")
# ------------------------选择数据是视频还是相机--------------------
mode = "video_test"
# mode = "camera"
comside = 'red'
comside = 'blue'
# -----------------------------相机----------------------------------------------------
if mode == "camera":
                     # cv::CAP_FFMPEG或cv::CAP_IMAGES或cv::CAP_DSHOW
    forward_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Util.final_black_camera(forward_camera)
    Util.set_laptop_camera(forward_camera)
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
    # forward_camera = cv2.VideoCapture("record_image/对国科大2022.6.16record.mp4")
    forward_camera = cv2.VideoCapture("record_image/广工test.mp4")

    size_y = forward_camera.get(4)#后续图像处理使用
    # 测试用,查看视频size
    size = (int(forward_camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(forward_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('size:' + repr(size))
# --------------------------------------------------------------------------------------
# ------------------------------场地缩略图------------------------------------------------
completesite = cv2.imread("images/competesite.png")
completesite = cv2.resize(completesite, (500, 1000), interpolation=cv2.INTER_CUBIC) #蓝方
if comside == 'red':
    completesite = cv2.flip(completesite,-1) #红方
Util.completesite_frame(completesite)
completesite_origin = completesite.copy()
# ----------------------------神经网络准备-------------------------------------------------
# yolo = YOLO()
#     # ----------------------------------------------------------------------------------------------------------#
#     #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
#     #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
#     #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
#     #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
#     #   video_fps用于保存的视频的fps
#     #   video_path、video_save_path和video_fps仅在mode='video'时有效
#     #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
#     # ----------------------------------------------------------------------------------------------------------#
# video_path = "2021-0801-1057-小组赛-四川大学VS东莞理工-RD0_Trim.mp4"
# video_save_path = ""
# video_fps = 25.0
#
# if video_save_path != '':
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     size = (int(forward_camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(forward_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

# -------------------------------------神经网络参数设置-----------------------------------------------
num_classes = 12
input_shape = (640, 640)
strides = torch.tensor([[32, 32], [16, 16], [8, 8]])
load_w_path= 'model_data/w760.pt'
classes_path = 'model_data/jiaban_classes2.txt'
phi='tiny'
classes=get_class(classes_path)
print('model run on ' + str(onnxruntime.get_device()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------onnx参数设置及初始化--------------------------------------------
# 静态运行onnxruntime，去掉初始化时间
print(onnxruntime.get_available_providers())
sess_options = onnxruntime.SessionOptions()
# # Set graph optimization level
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# # To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "onnxfiles/w760.ort"
onnx_session = onnxruntime.InferenceSession('onnxfiles/w760.onnx', providers=['CUDAExecutionProvider'], sess_options = sess_options)
# sess_options.optimized_model_filepath = "../serialed_trt_model/optimized_model.trt"
# onnx_session = onnxruntime.InferenceSession('onnxfiles/upup32.onnx', providers=['TensorrtExecutionProvider'], sess_options = sess_options)
# sess_options.optimized_model_filepath = "onnxfiles/???/w760.trt"
# onnx_session = onnxruntime.InferenceSession('onnxfiles/w760sim.onnx', providers=['TensorrtExecutionProvider'], sess_options = sess_options)
# ['CUDAExecutionProvider', 'CPUExecutionProvider']
input_name = onnx_session.get_inputs()[0].name
output_name = [onnx_session.get_outputs()[0].name, onnx_session.get_outputs()[1].name,
               onnx_session.get_outputs()[2].name]
# ----------------------------------------------------------------------------------------------
# 高斯模糊核
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((5, 5), np.uint8)
background = None
a=0
# ----------------------------------------标定----------------------------------------------
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
        # xy = "%d,%d" % (x, y)
        # xy = "%d,%d" % (g_location_win[0], g_location_win[1])
        cv2.circle(g_image_show, (x, y), 1, (255, 0, 0), thickness=2)
        cv2.putText(g_image_show, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), thickness = 1)
        point_reverscircle_list.append([x_o, y_o])
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
        g_image_zoom = cv2.resize(g_image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
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
# -----------------------------------------------------------------------------------------------------------

while True:
    alltime1 = time.time()
    cameratime1 = time.time()
    # 重载场地缩略图
    completesite = completesite_origin.copy()
    # 读取视频流
    grabbed, frame_lwpCV = forward_camera.read()
    if grabbed == False:
        continue
    # 对帧进行预处理，先转灰度图，再进行高斯滤波。
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。

    frame_lwpCV = cv2.resize(frame_lwpCV, (1250, 760), interpolation=cv2.INTER_CUBIC)
    image_shape = (frame_lwpCV.shape[1], frame_lwpCV.shape[0])
    cameratime2 = time.time()
    # ----------------------------------动态检测准备--------------------------------------------------
    try:
        gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    except:
        break
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray_lwpCV
        continue
    # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）
    diff= Util.diff(background, gray_lwpCV, es)
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
    # ---------------------------------神经网络预测追踪英雄位置-----------------------------------------

    networktime1 = time.time()

    image = frame_lwpCV.copy()
    image = image[:, :, ::-1]

    new_image = cv2.resize(np.array(image), (input_shape[0], input_shape[1]))
    new_image = new_image.astype('float') / 255.0
    new_image = np.transpose(new_image, (2, 0, 1))
    image_data = torch.from_numpy(new_image)
    image_data = image_data.float()
    input = image_data.view(1, 3, input_shape[1], input_shape[0]).numpy()

    outputs = onnx_session.run([output_name[0],output_name[1],output_name[2]], {input_name: input.astype(np.float32)})

    networktime2 = time.time()

    boxes_, classes_ = yolo_eval([torch.tensor(outputs[0]).to(device),torch.tensor(outputs[1]).to(device),torch.tensor(outputs[2]).to(device)],
                                 strides,
                                 input_shape,
                                 num_classes,
                                 image_shape,
                                 score_threshold=0.6)

    for j in range(len(boxes_)):
        boxes_ = boxes_.to(device)
        x1 = boxes_[j][..., 0].item()
        y1 = boxes_[j][..., 1].item()
        x2 = boxes_[j][..., 2].item()
        y2 = boxes_[j][..., 3].item()

        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        x_center = x_center
        y_center = y_center
        b = np.array([x_center, y_center], dtype=np.int32)
        c = classes_[j]
        c = classes[c]
        str = c
        a1 = np.array([x1, y1], dtype=np.int32)
        a2 = np.array([x2, y2], dtype=np.int32)
        frame_lwpCV = cv2.rectangle(frame_lwpCV, a1, a2, (0, 255, 0), 4)
        frame_lwpCV = cv2.rectangle(frame_lwpCV, b, b, (0, 255, 0), 4)
        cv2.putText(frame_lwpCV, str, b, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

    networktime3 = time.time()
    # -------------------------------------------------------------------------------------------
    # ----------------------------------飞坡预警--------能量机关预警---------------------------------
    warningtime1 = time.time()

    # 快速通过标框得到感兴趣区域坐标范围
    # (x1, y1, x2, y2, x3, y3, x4, y4) = param.get_Flyslope_param()
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = point_reverscircle_list[8:12]
    fpointrange = (x1, y1, x2, y2, x3, y3, x4, y4)
    cv2.polylines(frame_lwpCV, np.array([[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]]),1 ,(0,255,0),5,8,0)
    # (x5, y5, x6, y6, x7, y7, x8, y8) = param.get_energyorgan_param()
    [[x5, y5], [x6, y6], [x7, y7], [x8, y8]] = point_reverscircle_list[12:]
    epointrange = (x5, y5, x6, y6, x7, y7, x8, y8)
    cv2.polylines(frame_lwpCV, np.array([[[x5, y5], [x6, y6], [x7, y7], [x8, y8]]]), 1, (0, 255, 0), 5, 8, 0)
    (u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6, u7, v7, u8, v8, u9, v9) = param.get_site_param()

    # 框大小阈值
    area=5
    # 画面左上角为坐标原点, xup,yup,左上角点的坐标,xdown,ydown,右下角点的坐标
    for c in contours:
        # print(contours)
        if cv2.contourArea(c) < area:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        # 设置感兴趣区，只对这部分内出现的移动物体做标记
        if deter_point(fpointrange, x, y, size_y) and deter_point(fpointrange, x + w, y + h, size_y):
            print('---------------------------------------------------------------------------------------------------')
            cv2.fillPoly(completesite, np.array([[[u1, v1], [u2, v2], [u3, v3], [u4, v4]]]), (0, 0, 255), cv2.LINE_AA)
            cv2.fillPoly(frame_lwpCV, np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]), (0, 0, 255), cv2.LINE_AA)
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 传递飞坡车出现的信号
            # serial.proto_FLyslope_serial("飞坡预警",ser)

        if deter_point(epointrange, x, y, size_y) and deter_point(epointrange, x + w, y + h, size_y):
            cv2.fillPoly(completesite, np.array([[[u5, v5], [u6, v6], [u7, v7], [u8, v8], [u9, v9]]]), (0, 0, 255), cv2.LINE_AA)
            cv2.fillPoly(frame_lwpCV, np.array([[[x5, y5], [x6, y6], [x7, y7], [x8, y8]]]), (0, 0, 255), cv2.LINE_AA)
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)

    warningtime2 = time.time()
    # -------------------------------------------------------------------------------------------
    # ---------------------------------从裁判系统端读取车血量------------------------------------------
    # 由serial.py负责串口任务
    # ---------------------------------------------------------------------------------------------
    # -----------------------------------虚拟坐标--------------------------------------------------
    cortime1 = time.time()
    k1 = leftedge_to_Rbase/Bbase_to_Rbase
    k2 = leftedge_to_Radar/Bbase_to_Rbase
    Lbase_points = point_reverscircle_list[:4]
    Rbase_points = point_reverscircle_list[4:8]
    Lbase_T = coordinate(base_objpoints, Lbase_points)
    Rbase_T = coordinate(base_objpoints, Rbase_points)
    Radia_Lbase = [Lbase_T[0][0],Lbase_T[1][0]]
    Radia_Rbase = [Rbase_T[0][0],Rbase_T[1][0]]
    O_base_R = (Lbase_T[0] + k2 * (Rbase_T[0] - Lbase_T[0]), Lbase_T[1])
    O_base_L = (Rbase_T[0] + k2 * (Rbase_T[0] - Lbase_T[0]), Rbase_T[1])
    # O__是O‘
    O__base_R = (O_base_R[0], Lbase_T[1])
    O__base_L = (O_base_L[0], Rbase_T[1])
    outpost_pic_L = (110, 403)
    outpost_pic_R = (387, 594)
    Ksc = outpost_pic_R[1]/Rbase_T[1]

    # 预测后得到的车的位置转换到场地缩略图之后的位置
    # pridict_x = coordinate()
    # pridict_y = coordinate()
    # cv2.circle(completesite, (int(pridict_x), int(pridict_y)), 1, (255, 0, 0), thickness=-1)
    # cv2.putText(completesite, '1'# vehicle_class
    #             , (int(pridict_x), int(pridict_y)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)

    cv2.circle(completesite, (int(Ksc*Radia_Lbase[0]), int(Ksc*Radia_Lbase[1])), 1, (0, 255, 0), thickness=4)
    cv2.putText(completesite, '左前哨站'  # vehicle_class
                , (int(Ksc*Radia_Lbase[0]), int(Ksc*Radia_Lbase[1])), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=4)
    cv2.circle(completesite, (int(Ksc * Radia_Rbase[0]), int(Ksc * Radia_Rbase[1])), 1, (0, 255, 0), thickness=4)
    cv2.putText(completesite, '右前哨站'  # vehicle_class
                , (int(Ksc * Radia_Rbase[0]), int(Ksc * Radia_Rbase[1])), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0),
                thickness=4)
    cortime2 = time.time()
    # -------------------------------------------------------------------------------------------

    fps = 1 / (time.time() - alltime1)
    cv2.putText(frame_lwpCV, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 操作后重新设置backgrund，使上一帧图像作为下一帧图像的背景
    background = gray_lwpCV

    Util.cpsite_show('completesite', completesite)
    Util.detect_show('contours', frame_lwpCV)

    alltime2 = time.time()

    try:
        print("相机获取图像帧率", 1 / (cameratime2 - cameratime1), "time", cameratime2 - cameratime1)
        print("网络帧率", 1 / (networktime2 - networktime1), "time", networktime2 - networktime1)
        print("总帧率", 1/ (alltime2 - alltime1))
        print("网络后处理帧率", 1 / (networktime3 - networktime2), "time", networktime3 - networktime2)
        print("飞坡能量机关帧率", 1 / (float(warningtime2) - float(warningtime1)), "time", warningtime2 - warningtime1)
        print("坐标系转换帧率", 1 / (cortime2 - cortime1), "time", cortime2 - cortime1)
    except:
        pass

    # cv2.imshow('diff', diff)  #测试图像变化位置的灰度显示
    key = cv2.waitKey(1) & 0xFF
    # 按' '健退出循环
    if key == ord(' '):
        break

# 关闭串口，相机，opencv窗口
# ser.close()
forward_camera.release()
cv2.destroyAllWindows()
# out.release()
