#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/31 2:25
@Author:   sml2h3
@File:     server
@Software: PyCharm
"""
import io
import os
import json
import time
import onnx
import torchvision
import onnxruntime
from PIL import Image
import numpy as np
from config import Config


class Server(object):
    def __init__(self, project_name):
        self.project_name = project_name
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects",
                                      self.project_name)
        self.graph_path = os.path.join(self.base_path, 'graphs', '{}.onnx'.format(self.project_name))
        self.ort_session = onnxruntime.InferenceSession(self.graph_path)
        self.config = Config(project_name).load_config()
        self.charset = json.loads(self.config['Model']['CharSet'])

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def classification(self, img: bytes):
        image = Image.open(io.BytesIO(img)).convert('L')
        totensor = torchvision.transforms.ToTensor()
        resize = torchvision.transforms.Resize((150, 50))
        normalize = torchvision.transforms.Normalize((0.5), (0.5))
        image = resize(image)
        image = totensor(image)
        image = normalize(image)
        ort_inputs = {'input1': np.array([self.to_numpy(image)])}
        ort_outs = self.ort_session.run(None, ort_inputs)
        result = []
        for item in ort_outs[0][0]:
            if item != 0:
                result.append(self.charset[item])
        return result


server = Server('PROJECT_NAME')
with open(r"PATH TO YOUR IMAGE", 'rb') as f:
    img = f.read()
start_time = int(time.time() * 1000)
r = server.classification(img)
print(int(time.time() * 1000) - start_time)
print(r)