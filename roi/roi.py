import time
import copy
import numpy as np
import os
import cv2
import onnxruntime
import torch
# 静态运行onnxruntime，去掉初始化时间
print(onnxruntime.get_available_providers())
sess_options = onnxruntime.SessionOptions()
sess_options.optimized_model_filepath = "abc.ort"
onnx_session = onnxruntime.InferenceSession('yolo1260.onnx', providers=['CUDAExecutionProvider'], sess_options = sess_options)
# onnx_session = onnxruntime.InferenceSession('upupup.onnx', providers=['CUDAExecutionProvider'])
# ['CUDAExecutionProvider', 'CPUExecutionProvider']
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name
filepath = '../images/night/MER-133-54U3C-L(FH0190030019)'
IOUThreshold = 0.1


class Point:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
    def fixpoint(self):
        self.x = self.x / 640 * 1280
        self.y = self.y / 384 * 960
    def point(self):
        return [int(self.x),int(self.y)]

class Armor:
    def __init__(self,lt=[0,0], lb=[0,0], rt=[0,0], rb=[0,0]):
        self.leftTop = Point(lt[0], lt[1])
        self.leftBottom = Point(lb[0], lb[1])
        self.rightTop = Point(rt[0], rt[1])
        self.rightBottom = Point(rb[0], rb[1])
        self.color = str()
        self.confidence = 0

    def set(self, wlt, wlb, wrt, wrb):
        lt = wlt
        lb = wlb
        rt = wrt
        rb = wrb
        self.leftTop = Point(lt[0], lt[1])
        self.leftBottom = Point(lb[0], lb[1])
        self.rightTop = Point(rt[0], rt[1])
        self.rightBottom = Point(rb[0], rb[1])
    def lt(self, lt: list):
        self.leftTop = Point(lt[0], lt[1])
    def lb(self, lb):
        self.leftBottom = Point(lb[0], lb[1])
    def rt(self, rt):
        self.rightTop = Point(rt[0], rt[1])
    def rb(self, rb):
        self.rightBottom = Point(rb[0], rb[1])

    def __lt__(self, other):
        if isinstance(self, Armor):
            return self.confidence < other.confidence
        else:
            print('not a Armor')
    def __gt__(self, other):
        if isinstance(self, Armor):
            return self.confidence > other.confidence
        else:
            print('not a Armor')

'''
    img: 输入的图像，指的是神经网络的输入图像，与input_node_dims对应（准确来说是存入Armor的坐标代表的图像大小）
    armor: 将网络解码后得到的坐标实例化并存入Armor中
'''

def get_roi(img, armor, input_node_dims, srcimg_size):
    y_extend_scale = 7. / 13.
    x_extend_scale = 1.7 / 10.
    x_max = min(max(max(max(armor.leftTop.x, armor.leftBottom.x), armor.rightTop.x), armor.rightBottom.x),
                srcimg_size[0])
    y_max = min(max(max(max(armor.leftTop.y, armor.leftBottom.y), armor.rightTop.y), armor.rightBottom.y),
                srcimg_size[1])
    x_min = max(min(min(min(armor.leftTop.x, armor.leftBottom.x), armor.rightTop.x), armor.rightBottom.x), 0)
    y_min = max(min(min(min(armor.leftTop.y, armor.leftBottom.y), armor.rightTop.y), armor.rightBottom.y), 0)
    if x_max + ((x_max - x_min) * x_extend_scale / 2) < srcimg_size[0]:
        x_max = x_max + ((x_max - x_min) * x_extend_scale / 2.)
    else:
        x_max = srcimg_size[0]
    if y_max + ((y_max - y_min) * y_extend_scale / 2) < srcimg_size[1]:
        y_max = y_max + ((y_max - y_min) * y_extend_scale / 2.)
    else:
        y_max = srcimg_size[1]
    if x_min - ((x_max - x_min) * x_extend_scale / 2) > 0:
        x_min = x_min - ((x_max - x_min) * x_extend_scale / 2.)
    else:
        x_min = 0
    if y_min - ((y_max - y_min) * y_extend_scale / 2) > 0:
        y_min = y_min - ((y_max - y_min) * y_extend_scale / 2.)
    else:
        y_min = 0
    # x_min, y_min, x_max, y_max
    roi_range = [int(x_min), int(y_min), int(x_max), int(y_max)]
    roi = img[roi_range[1]:roi_range[3], roi_range[0]:roi_range[2], :]
    cv2.imshow("roi", roi)
    cv2.waitKey()
    return roi

def encoding(output):
    armorlist = []
    for i in range(1260):
        if output[0, i ,8] > 0.9:
            armor = Armor()
            armor.confidence = output[0, i ,8]
            print(armor.confidence)
            armor.lt([output[0, i, 0], output[0, i, 1]])
            armor.leftTop.fixpoint()
            armor.lb([output[0, i, 2], output[0, i, 3]])
            armor.leftBottom.fixpoint()
            armor.rb([output[0, i, 4], output[0, i, 5]])
            armor.rightBottom.fixpoint()
            armor.rt([output[0, i, 6], output[0, i, 7]])
            armor.rightTop.fixpoint()
            armorlist.append(armor)
    print('detected  ' + str(len(armorlist)) + '  armor')
    return armorlist

def get_mask(armor: Armor):
    img_mask = np.zeros((1280, 960))
    rec_arr = np.array([armor.leftTop.point(), armor.rightTop.point(), armor.rightBottom.point(), armor.leftBottom.point()]).astype(np.int32)
    img_mask = cv2.fillPoly(img_mask, [rec_arr], 1)
    img_mask = img_mask.astype(np.uint8)
    return img_mask

def getIOU(armor1: Armor, armor2: Armor):
    # InterSectionArea = cv2.rotatedRectangleIntersection(((armor1.leftTop.x, armor1.leftTop.y), (armor1.rightBottom.x, armor1.rightBottom.y)),
    #                                                     ((armor2.leftTop.x, armor2.leftTop.y), (armor2.rightBottom.x, armor2.rightBottom.y)))
    mask1 = get_mask(armor1)
    mask2 = get_mask(armor2)
    intersection = mask1 * mask2
    interarea = np.sum(intersection[intersection == 1])
    if(interarea <= 0):
        return 0;
    return 10;

def NMS(armorlist: list):
    armorlist.sort()
    removed = [False for i in range(2500)]
    for i in range(len(armorlist)):
        if removed[i]:
            continue
        for j in range(i + 1, len(armorlist)):
            if removed[j]:
                continue
            if (getIOU(armorlist[i], armorlist[j]) >= IOUThreshold):
                removed[j] = True
                armorlist[i].confidence = 0
    for armor in armorlist:
        if armor.confidence == 0:
            armorlist.remove(armor)
    return armorlist

for root, dir, files in os.walk(filepath):
    root, dir, files = root, dir, files
index = 0
for imgpath in files:
    index = index + 1
    img = cv2.imread(root + '/' + imgpath)
    srcimg = img.copy()
    img = img.copy()
    img = cv2.resize(np.array(img),(640, 384))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    # img = img.astype(np.uint8)
    # cv2.imshow("1q2e", img[0])
    # cv2.waitKey()

    time1 = time.time()
    output = onnx_session.run([output_name], {input_name: img.astype(np.float32)})
    output = output[0]
    time2 = time.time()
    fps = 1/(time2 - time1)

    armorlist = encoding(output)
    armorlist = NMS(armorlist)

    print('final  ' + str(len(armorlist)) + '  armor')
    for armor in armorlist:
        roi = get_roi(srcimg, armor, [1, 384, 640, 3], [1280, 960])
        cv2.imwrite("../images/roi/"+str(index)+'.jpg', roi)
        # cv2.line(srcimg, tuple(armor.leftTop.point()), tuple(armor.rightBottom.point()), (0, 255, 0))
        # cv2.line(srcimg, tuple(armor.rightBottom.point()), tuple(armor.leftBottom.point()), (0, 255, 0))
        # cv2.line(srcimg, tuple(armor.leftBottom.point()), tuple(armor.rightTop.point()), (0, 255, 0))
        # cv2.line(srcimg, tuple(armor.leftTop.point()), tuple(armor.rightTop.point()), (0, 255, 0))
        # cv2.imshow('img', srcimg)
        # cv2.waitKey()


# import torch
# import matplotlib.pyplot as plt
# xs = torch.linspace(-5, 5, steps=100)
# ys = torch.linspace(-5, 5, steps=100)
# x, y = torch.meshgrid(xs, ys, indexing='xy')
# z = torch.sin(torch.sqrt(x * x + y * y))
# ax = plt.axes(projection='3d')
# ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
# plt.show()
