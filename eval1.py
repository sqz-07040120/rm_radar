#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from PIL import Image
import numpy as np
import cv2
import time
from nets.yolo import YoloBody
import torchvision.ops as ops
from utils import Util
# function--------------------------------------------------------------------------
def get_class(classes_path):
    with open(classes_path) as f:
        classes = f.readline()
    classes = [str(x) for x in classes.split(',')]
    return np.array(classes)

def yolo_head(output, stride, input_shape, image_shape, device):
    input_shape = torch.tensor(input_shape).float().to(device)
    image_shape = torch.tensor(image_shape).float().to(device)
    stride = stride.to(device)

    hsize, wsize = output.shape[-2:]
    yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
    grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
    grid = grid.view(1, -1, 2)
    output = output.flatten(start_dim=2).permute(0, 2, 1)
    output[..., :2] = (output[..., :2] + grid) * stride * image_shape / input_shape
    output[..., 2:4] = torch.exp(output[..., 2:4]) * stride * image_shape / input_shape

    x1 = output[..., 0:1] - output[..., 2:3] / 2
    y1 = output[..., 1:2] - output[..., 3:4] / 2
    x2 = output[..., 0:1] + output[..., 2:3] / 2
    y2 = output[..., 1:2] + output[..., 3:4] / 2

    output[..., 0:1] = x1
    output[..., 1:2] = y1
    output[..., 2:3] = x2
    output[..., 3:4] = y2
    output[..., 4:5] = torch.sigmoid(output[..., 4:5])
    output[..., 5: ] = torch.sigmoid(output[..., 5:])

    return output

# def yolo_eval(yolo_outputs,
#               strides,
#               input_shape,
#               num_classes,
#               image_shape,
#               score_threshold=0.6,
#               iou_threshold=0.5):
#     while True:
#         outputs = []
#         for l in range(len(yolo_outputs)):
#             output = yolo_head(yolo_outputs[l],strides[l], input_shape, image_shape)
#             outputs.append(output)
#         outputs = torch.cat(outputs, dim=1)
#         boxes = outputs[..., 0:4]
#         box_scores = outputs[..., 4:5] * outputs[..., 5:]
#         mask = box_scores >= score_threshold
#         boxes_ = []
#         classes_ = []
#         for c in range(num_classes):
#             mask_c = mask[...,c:c+1].repeat(1,1,4)
#             class_box_scores=torch.masked_select(box_scores[...,c],mask[...,c])
#             class_boxes=torch.masked_select(boxes,mask_c)
#             class_boxes=class_boxes.view(-1,4)
#             class_box_scores=class_box_scores.view(-1)
#             #返回nms的引索
#             index = ops.nms(boxes=class_boxes, scores=class_box_scores, iou_threshold=iou_threshold)
#             class_box_scores = torch.gather(class_box_scores,0,index)
#             index = index.cpu().numpy().tolist()
#             class_boxes = class_boxes[index]
#             ##给符合要求的框标类
#             classes = torch.ones_like(class_box_scores, dtype=torch.int32) * c
#             boxes_.append(class_boxes)
#             classes_.append(classes)
#         if boxes_ == []:
#             return boxes_, classes_
#         else:
#             boxes_ = torch.cat(boxes_, dim=0)
#             boxes_ = boxes_.view(-1,4)
#             classes_= torch.cat(classes_,dim=-1)
#             return boxes_, classes_

def yolo_eval(yolo_outputs,
              strides,
              input_shape,
              num_classes,
              image_shape,
              score_threshold=0.6):
    outputs = []
    for l in range(len(yolo_outputs)):
        output = yolo_head(yolo_outputs[l],strides[l], input_shape, image_shape, 'cuda')
        outputs.append(output)
    outputs = torch.cat(outputs, dim=1)
    boxes = outputs[..., 0:4]
    box_scores = outputs[..., 4:5] * outputs[..., 5:]
    mask = box_scores >= score_threshold
    boxes_ = []
    classes_ = []
    for c in range(num_classes):
        mask_c = mask[...,c:c+1].repeat(1,1,4)
        class_box_scores=torch.masked_select(box_scores[...,c],mask[...,c])
        class_boxes=torch.masked_select(boxes,mask_c)
        class_boxes=class_boxes.view(-1,4)
        class_box_scores=class_box_scores.view(-1)
        if class_box_scores.size()[0]==0:
            continue
        #返回max的引索
        index = torch.argmax(class_box_scores)
        class_box_scores = torch.gather(class_box_scores,0,index)
        index = index.cpu().numpy().tolist()
        class_boxes = class_boxes[index]
        ##给符合要求的框标类
        classes = torch.ones_like(class_box_scores, dtype=torch.int32) * c
        boxes_.append(class_boxes)
        classes_.append(classes)
    if boxes_ == []:
        return boxes_, classes_
    else:
        boxes_ = torch.cat(boxes_, dim=0)
        boxes_ = boxes_.view(-1,4)
        return boxes_, classes_

# if __name__ == "__main__":
#     # 参数设置-----------------------------------------------------------------------------------------
#     num_classes = 12
#     input_shape = (640, 640)
#     strides = torch.tensor([[32, 32], [16, 16], [8, 8]])
#     #image_shape = (1920, 1080)
#     #image_shape = (1808,1080)
#     #image_shape = (3456,2160)
#
#     range_start = 50
#     range_stop = 200
#     load_w_path= 'model_data/_w290.pt'
#     path1='images/11111.png'
#     # path2='C:\\Users\\imika\\Desktop\\labels\\'
#     forward_camera = cv2.VideoCapture("images/rm上交雷达左视角.mp4")
#     classes_path = 'model_data/jiaban_classes2.txt'
#     phi='tiny'
#     # run----------------------------------------------------------------------------------------------
#     classes=get_class(classes_path)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net2 = YoloBody(num_classes, phi)
#     net2.to(device)
#     net2.load_state_dict(torch.load(load_w_path))
# #-----------------------------------------------------------------------------------------------------
#     '''导出onxx文件'''
#
#     # dummy_input1 = torch.randn(1, 3, 640, 640,requires_grad=True).cuda()
#     # torch.onnx.export(net2, dummy_input1, "try2.onnx", verbose=True)
# #-----------------------------------------------------------------------------------------------------
#
#     while True:
#
#         grabbed, frame_lwpCV = forward_camera.read()
#
#         frame_lwpCV = cv2.resize(frame_lwpCV, (1250, 760), interpolation=cv2.INTER_CUBIC)
#         image_shape = (frame_lwpCV.shape[1], frame_lwpCV.shape[0])
#         #flag = os.path.exists(os.path.join(path2)+ '%d.txt' % i)
#
#         #image2 = cv2.imread('D:\\DJI_ROCO\\robomaster\\image\\%d.jpg' % i)
#         #image2 = cv2.imread('C:\\Users\\imika\\Pictures\\sj\\%d.jpg' % i)
#         # image2 = cv2.imread(path1)
#         image2 = frame_lwpCV
#
#         image = image2.copy()
#         image = image[:, :, ::-1]
#
#         new_image=cv2.resize(np.array(image), (input_shape[0], input_shape[1]))
#         new_image = new_image.astype('float') / 255.0
#         new_image = np.transpose(new_image, (2, 0, 1))
#         image_data = torch.from_numpy(new_image)
#         image_data = image_data.float()
#         input = image_data.view(1, 3, input_shape[1], input_shape[0])
#         input = input.to(device)
#         since = time.time()
#         outputs = net2(input)
#         time1 = time.time() - since
#         print("time_model:", time1)
#         boxes_, classes_= yolo_eval(outputs,
#                       strides,
#                       input_shape,
#                       num_classes,
#                       image_shape,
#                       score_threshold=0.6)
#         time2 = time.time() - since
#         print("time_all:", time2)
#         n = len(boxes_)
#         if n ==0:
#             cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
#             cv2.imshow('input_image', image2)
#             cv2.waitKey(1)
#
#         for j in range(n):
#             boxes_ = boxes_.cpu()
#             x1 = boxes_[j][..., 0].item()
#             y1 = boxes_[j][..., 1].item()
#             x2 = boxes_[j][..., 2].item()
#             y2 = boxes_[j][..., 3].item()
#
#             x_center = (x1 + x2) // 2
#             y_center = (y1 + y2) // 2
#             x_center = x_center
#             y_center = y_center
#             b = np.array([x_center,y_center],dtype=np.int32)
#             c = classes_[j]
#             c = classes[c]
#             str = c
#             a1 = np.array([x1, y1], dtype=np.int32)
#             a2 = np.array([x2, y2], dtype=np.int32)
#             image2 = cv2.rectangle(image2, a1, a2, (0, 255, 0), 4)
#             image2 = cv2.rectangle(image2, b, b, (0, 255, 0), 4)
#             cv2.putText(image2, str, b, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
# # 打印label-----------------------------------------------------------------------------------------
# #         if flag is True:
# #             print(i)
# #             data = np.loadtxt(os.path.join(path2)+ '%d.txt' % i, dtype=np.float32, delimiter=' ')
# #             data = data.reshape(-1,5)
# #             for i in range(data.shape[0]):
# #                 x1 = data[i][1]
# #                 y1 = data[i][2]
# #                 x2 = data[i][3]
# #                 y2 = data[i][4]
# #                 x_center = (x1 + x2) // 2
# #                 y_center = (y1 + y2) // 2
# #                 b = np.array([x_center, y_center], dtype=np.int32)
# #                 c = int(data[i][0])
# #                 c = classes[c]
# #                 str = c
# #                 a1 = np.array([x1, y1], dtype=np.int32)
# #                 a2 = np.array([x2, y2], dtype=np.int32)
# #                 image2 = cv2.rectangle(image2, a1, a2, (255, 0, 0), 1)
# #                 image2 = cv2.rectangle(image2, b, b, (255, 0, 0), 1)
# #                 cv2.putText(image2, str, b, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
# # -----------------------------------------------------------------------------------------
# #         cv2.namedWindow('input_image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)#, cv2.WINDOW_AUTOSIZE
#         Util.detect_show('input_image', frame_lwpCV)
#         # cv2.imshow('input_image', image2)
#         cv2.waitKey(1)