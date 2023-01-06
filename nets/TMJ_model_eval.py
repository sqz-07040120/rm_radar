import torch
from PIL import Image
import model
import numpy as np
import cv2
from qwqtorch import orz_model
import reloss
import os
from torch.utils.data import Dataset, DataLoader
from gen_data import MyDataset
import time

# function--------------------------------------------------------------------------
def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例
    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 采用双三次插值算法缩小图像
    # image.show()
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # new_image.show()
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    # new_image.show()
    return new_image

def get_class(classes_path):
    with open(classes_path) as f:
        classes = f.readline()
    classes = [str(x) for x in classes.split(',')]
    return np.array(classes)

def yolo_head(feats, num_classes, num_teams, input_shape, image_shape):
    grid_shape = [list((feats).size())[1:3]]
    grid_shape = np.array(grid_shape, dtype='int').reshape(2,)
    grid_y = np.tile(np.reshape(np.arange(0, grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    grid = np.repeat(grid, 4, axis=2)
    grid = grid.reshape(1, grid_shape[0], grid_shape[1], 1, 8)
    grid = torch.from_numpy(grid)
    grid = grid.float()

    feats = feats.view(-1, grid_shape[0], grid_shape[1], 1, num_classes + num_teams + 9)

    grid_shape = np.repeat(grid_shape, 4, axis=0)
    grid_shape = torch.from_numpy(grid_shape)
    grid = grid.to(device)
    grid_shape = grid_shape.to(device)
    box_xy = (feats[..., :8] + grid) / grid_shape
    box_confidence = torch.sigmoid(feats[..., 8:9])
    box_team_probs = torch.sigmoid(feats[..., 9:13])
    box_class_probs = torch.sigmoid(feats[..., 13:])

    input_shape = torch.tensor(input_shape)
    image_shape = torch.tensor(image_shape)
    input_shape = input_shape.float()
    image_shape = image_shape.float()
    input_shape = input_shape.to(device)
    image_shape = image_shape.to(device)

    new_shape = torch.round(image_shape * torch.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_xy = box_xy.view(-1, grid_shape[0], grid_shape[1], 4, 2)
    box_xy = (box_xy- offset) * scale * image_shape
    box_xy = box_xy.view(-1, grid_shape[0], grid_shape[1], 1, 8)
    return box_xy, box_confidence, box_team_probs, box_class_probs

def yolo_boxes_and_scores(feats,num_classes, input_shape, image_shape):
    boxes, box_confidence, box_team_probs, box_class_probs = yolo_head(feats, num_classes, num_teams, input_shape,
                                                                       image_shape)
    boxes = boxes.view(-1, 8)
    confidence_scores = box_confidence.view(-1,1)
    team_scores = box_team_probs.view(-1, 4)
    class_scores = box_class_probs.view(-1, 9)
    return boxes, confidence_scores, team_scores, class_scores


def yolo_eval(yolo_outputs,
              input_shape,
              num_classes,
              image_shape,
              score_threshold=0.6):
    while True:
        num_layers = len(yolo_outputs)
        boxes = []
        scores = []
        teams = []
        classes = []
        for l in range(num_layers):
            _boxes, confidence_scores, team_scores, class_scores = yolo_boxes_and_scores(yolo_outputs[l],num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            scores.append(confidence_scores)
            teams.append(team_scores)
            classes.append(class_scores)
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        teams = torch.cat(teams, dim=0)
        classes = torch.cat(classes, dim=0)
        mask = scores >= score_threshold

        class_boxes=[]
        class_box_teams=[]
        class_box_classes=[]
        for i in range(8400):
            if mask[i,0]==1:
                class_boxes.append(boxes[i])
                class_box_teams.append(teams[i])
                class_box_classes.append(classes[i])
        if class_boxes == []:
            return class_boxes,class_box_teams,class_box_classes
        class_boxes = torch.tensor([item.cpu().detach().numpy() for item in class_boxes]).cuda()
        class_box_teams = torch.tensor([item.cpu().detach().numpy() for item in class_box_teams]).cuda()
        class_box_classes = torch.tensor([item.cpu().detach().numpy() for item in class_box_classes]).cuda()
        boxes_ = class_boxes.int()
        teams_ = torch.argmax(class_box_teams, dim=-1)
        classes_ = torch.argmax(class_box_classes, dim=-1)
        return boxes_, teams_, classes_

if __name__ == "__main__":
    # 参数设置-------------------------------------------------------------------------
    num_layers = 3
    num_teams = 4
    num_classes = 9
    input_shape = (640, 640)
    image_shape = (1280, 1024)
    range_start = 4527
    range_stop = 4535
    lr = 0.0001
    image_path = 'images2/'
    classes_path = 'jiaban_classes.txt'
    # run----------------------------------------------------------------------------------------------
    since=time.time()
    classes=get_class(classes_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net2 = model.yolox(num_teams=num_teams,num_classes=num_classes)
    #net2 = orz_model.yolov5s(num_anchors=1,num_teams=num_teams,num_classes=num_classes)
    net2.to(device)
    net2.load_state_dict(torch.load('weight2.pt'))


    for i in range(range_start,range_stop):
        image2 = cv2.imread('images2/%d.jpg' % i)
        image = Image.open('images2/%d.jpg' % i)
        new_image = pad_image(image, input_shape)
        new_image = np.array(new_image)
        new_image = new_image.astype('float') / 255.0
        new_image = np.transpose(new_image, (2, 0, 1))
        image_data = torch.from_numpy(new_image)
        image_data = image_data.float()
        input = image_data.view(1, 3, 640, 640)
        input = input.to(device)
        output = net2(input)
        print(i)
        boxes_, teams_, classes_= yolo_eval(output,
                      input_shape,
                      num_classes,
                      image_shape,
                      score_threshold=0.6)
        time_elapsed = time.time() - since
        print("time:",time_elapsed)
        n = len(boxes_)
        if n ==0:
            cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('input_image', image2)
            cv2.waitKey(1)

        for j in range(n):
            x1 = boxes_[j][..., 0]
            y1 = boxes_[j][..., 1]
            x2 = boxes_[j][..., 2]
            y2 = boxes_[j][..., 3]
            x3 = boxes_[j][..., 4]
            y3 = boxes_[j][..., 5]
            x4 = boxes_[j][..., 6]
            y4 = boxes_[j][..., 7]

            x_center = (x1 + x2 + x3 + x4) // 4
            y_center = (y1 + y2 + y3 + y4) // 4
            x_center = x_center.cpu()
            y_center = y_center.cpu()
            b = np.array([x_center,y_center],dtype=np.int32)
            print(b)
            if torch.any(boxes_[j]) is True:
                continue
            c = teams_[j] * 9 + classes_[j]
            c = classes[c]
            str = c
            a = np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.int32).reshape(4, 2)
            image2 = cv2.polylines(image2, [a], True, (0, 255, 0), 1)
            image2 = cv2.rectangle(image2, b, b, (0, 255, 0), 1)
            cv2.putText(image2, str, b, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('input_image', image2)
        cv2.waitKey(1)