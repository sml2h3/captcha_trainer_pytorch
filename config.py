#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/23 18:08
@Author:   sml2h3
@File:     config
@Software: PyCharm
"""
from utils.constants import Labels
import os
import json
import yaml


class Config(object):

    def __init__(self, project_name):
        self.project_name = project_name
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")
        self.config_dict = {
            "System": {
                "Project": None,
                "GPU": True,
                "GPU_ID": 0
            },
            "Model": {
                "ImageWidth": 150,
                "ImageHeight": 50,
                "ImageChannel": 1,
                "RESIZE": [150, 50],
                "CharSet": json.dumps(Labels.NumbersAndAlphabet.value, ensure_ascii=False),
                "Word": False
            },
            "Train": {
                "BATCH_SIZE": 32,
                "TEST_BATCH_SIZE": 32,
                'LSTM': {
                    'HIDDEN_NUM': 64,
                    'DROPOUT': 0.8
                },
                'CNN': {
                    "NAME": "MobileNetV2",
                },
                'RNN': {
                    "NAME": "LSTM"
                },
                'OPTIMIZER': 'Momentum',
                "TEST_STEP": 1000,
                "TARGET": {
                    "Accuracy": 0.97,
                    "Epoch": 200,
                    "Cost": 0.005
                },
                "LR": 0.01
            }
        }

    def make_config(self, config_dict=None):
        self.config_dict["System"]["Project"] = self.project_name
        config_path = os.path.join(self.base_path, self.project_name, "config.yaml")
        with open(config_path, 'w', encoding="utf-8") as f:
            if config_dict is None:
                yaml.dump(self.config_dict, f, allow_unicode=True)
            else:
                yaml.dump(config_dict, f, allow_unicode=True)

    def load_config(self):
        config_path = os.path.join(self.base_path, self.project_name, "config.yaml")
        with open(config_path, 'r', encoding="utf-8") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict
