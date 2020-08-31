#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/23 0:11
@Author:   sml2h3
@File:     make_dataset
@Software: PyCharm
"""
from utils.constants import *
from utils.exception import *
from config import Config
from PIL import Image
import os
import sys
import json
import time
import random


class MakeDataset(object):
    def __init__(self, project_name: str, images_path: str, word: bool = False,
                 datatype: DataType = DataType.ClassFication):
        self.project_name = project_name
        self.images_path = images_path
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                      self.project_name)
        self.data_path = os.path.join(self.base_path, "datas")
        if not os.path.exists(self.data_path):
            raise DataDirNotFoundError(
                "Project named {} didn't found data dir!Please init project first!".format(self.project_name))
        self.word = word
        self.files_list = [os.path.join(self.images_path, filename) for filename in os.listdir(self.images_path)]
        self.label_list = self.get_labels()
        self.len = len(self.files_list)
        self.label_len = len(self.label_list)
        self.datatype = datatype
        self.ignore_files = ['.DS_Store']
        self.allow_ext = ['jpg', 'png', 'jpeg', 'bmp']

    def get_labels(self):
        label_map_temp = [""]
        for idx, filename in enumerate(self.files_list):
            path, labels = os.path.split(filename)
            labels = str(labels.split('_')[0]).lower()
            if self.word:
                label_map_temp += [labels]
            else:
                label_map_temp += list(labels)
        return list(set(label_map_temp))

    def make(self, scale: float = 0.97):
        random_seed = int(time.time() * 1000)
        random.seed(random_seed)
        random.shuffle(self.files_list)
        train_data_num = int(self.len * scale)
        test_data_num = self.len - train_data_num
        train_dataset = self.files_list[test_data_num:]
        test_dataset = self.files_list[:test_data_num]
        dataset = {
            RunMode.Train.value: train_dataset,
            RunMode.Test.value: test_dataset
        }
        for dataset_type in [RunMode.Train, RunMode.Test]:
            data_path = os.path.join(self.data_path, "{}_{}.json".format(self.project_name, dataset_type.value))
            if os.path.exists(data_path):
                os.remove(data_path)
            if dataset_type.value == RunMode.Train.value:
                used_dataset = dataset[RunMode.Train.value]
            else:
                used_dataset = dataset[RunMode.Test.value]
            self._covert_img_tojson(used_dataset, data_path)
        config = Config(self.project_name)
        conf = config.load_config()
        conf["Model"]["CharSet"] = json.dumps(self.label_list, ensure_ascii=False)
        config.make_config(conf)

    def _covert_img_tojson(self, dataset, output):
        simple_collection_length = len(dataset)
        collects = []
        for idx, filename in enumerate(dataset):
            if filename in self.ignore_files or filename.split('.')[-1].lower() not in self.allow_ext:
                continue
            else:
                try:
                    sys.stdout.write(
                        '\r{}'.format(">> Converting Image {}/{}".format(idx + 1, simple_collection_length)))
                    sys.stdout.flush()
                    Image.open(filename)

                    collects.append(filename)
                except Exception as e:
                    print(e)
        print("\n")
        with open(output, 'w', encoding="utf-8") as f:
            f.write(json.dumps(collects, ensure_ascii=False))
