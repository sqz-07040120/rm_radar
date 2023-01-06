from collections import defaultdict
import torch.nn as nn
import torch
import numpy as np

# --------------------------------------------官方用法示例--------------------------------------
# onnx_model = onnx.load("slitestedup.onnx")
# onnx.checker.check_model(onnx_model)
# ort_sess = ort.InferenceSession("slitestedup.onnx")
# outputs = ort_sess.run(None, {'input': 0})
# import onnxruntime
#
# # tensor = onnx.TensorProto()
# #     with open(input_file, 'rb') as f:
# #         tensor.ParseFromString(f.read())
# #         inputs.append(numpy_helper.to_array(tensor))
#
# # 创建一个InferenceSession的实例，并将模型的地址传递给该实例
# sess = onnxruntime.InferenceSession('slitestedup.onnx')
# # 调用实例sess的run方法进行推理
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy().astype(np.float32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float32)
# outputs = sess.run('output', {'input': to_numpy(dummy_input)})

# ------------------------------生成onnx------------------------------
from nets.yolo import YoloBody
import onnxruntime
import time
import cv2

# --------------------------------------------导出onnx-------------------------------------------
# num_classes = 12
# input_shape = (640, 640, 3)
# strides = torch.tensor([[32, 32], [16, 16], [8, 8]])
# load_w_path= '../model_data/w760.pt'
# phi='tiny'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net2 = YoloBody(num_classes, phi)
# net2.to(device)
# net2.load_state_dict(torch.load(load_w_path))
# net2.eval().to(device)
#
# new_image = cv2.resize(np.array(torch.randn(input_shape)), (input_shape[0], input_shape[1]))
# new_image = new_image.astype('float') / 255.0
# new_image = np.transpose(new_image, (2, 0, 1))
# image_data = torch.from_numpy(new_image)
# image_data = image_data.float()
# input = image_data.view(1, 3, input_shape[1], input_shape[0])
# input = input.to(device)
# torch.onnx.export(net2, input, 'onnxfiles/w760.onnx', verbose=False, opset_version=12 ,  input_names = ['input'], output_names = ['80time80','40time40','20time20'])

# -----------------------------------运行onnx-------------------------------------------
# onnx_session = onnxruntime.InferenceSession('slitestedup.onnx')
# # onnx_session = onnxruntime.InferenceSession('models/newpr/prtested.onnx')
# # inputs = (torch.randn(1,12,320,320),  torch.randn(1,12,320,320))
# inputs = np.ones((1,384,640,3), dtype=int)
# # inputs = (torch.randn(1,384,640,3),  torch.randn(1,384,640,3))
# output_name = onnx_session.get_outputs()[0].name
# input_name = onnx_session.get_inputs()[0].name
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy().astype(np.float32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float32)
# time1=time.time()
# res = onnx_session.run([output_name], {input_name:to_numpy(inputs[0])})[0]
# time2=time.time()
#
# print("1",time2-time1)
# print("2")

# from model.yolo_model_ import yolov5s_backbone, yolo5s_body
# import time
#
# net = yolo5s_body(num_anchor=1,num_team=4,num_class=9)
# net.load_state_dict(torch.load('slitestedup_model.pth',map_location='cpu'))
#
# dummy_input=torch.randn([1,3,640,640]).to('cpu')
#
# tic = time.time()
# net(dummy_input)
# print('time: ', time.time() - tic)

# ---------------------------onnx-simplifier 将原onnx中的int64参数转化为int32参数--------------
import onnx
from onnxsim import simplify

# onnx_model = onnx.load('upupup.onnx')
# model_simplified, check = simplify(onnx_model)
# assert check, "Simplified ONNX model could not be validated"
# onnx.save(model_simplified, 'upup32.onnx')

onnx_model = onnx.load('w760.onnx')
model_simplified, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simplified, 'w760sim.onnx')
print('finished exporting onnx')
