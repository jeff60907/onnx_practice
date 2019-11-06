# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:01:49 2019

@author: 07074
"""

import torch.onnx
import torch
import numpy as np

import onnx
import onnxruntime

Model_path = "./model.onnx"
input_path = "./img_7.jpg"

model = onnx.load(Model_path)
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))


ort_session = onnxruntime.InferenceSession(Model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = torch.randn(1, 1, 28, 28)
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)


from PIL import Image
import torchvision.transforms as transforms

img = Image.open(input_path).convert('1')

to_tensor = transforms.ToTensor()
img_y = to_tensor(img)
img_y.unsqueeze_(0)


ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)

answer = softmax(img_out_y)

print("predict : ", np.argmax(answer))