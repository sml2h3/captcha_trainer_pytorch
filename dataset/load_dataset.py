#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/23 0:11
@Author:   sml2h3
@File:     make_dataset
@Software: PyCharm
"""

from PIL import Image
from config import Config
from utils.constants import *
from utils.exception import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import json
import torch
import numpy as np
import torchvision


class LoadDataset(Dataset):

    def __init__(self, project_name, transform=None, mode=RunMode.Train):
        self.project_name = project_name
        self.transform = transform
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                      self.project_name)
        self.data_path = os.path.join(self.base_path, 'datas')
        with open(os.path.join(self.data_path, "{}_{}.json".format(self.project_name, mode.value)), 'r',
                  encoding="utf-8") as f:
            self.dataset = json.load(f)
        conf = Config(self.project_name)
        self.config = conf.load_config()
        self.charset = json.loads(self.config['Model']['CharSet'])
        self.word = self.config["Model"]["Word"]
        self.channel = self.config['Model']['ImageChannel']
        self.resize = self.config['Model']['RESIZE']
        self.label = []
        self.tensors = []
        for filename in self.dataset:
            self.tensors.append(filename)
            if self.word:
                # self.label.append([self.charset.index(filename.split('\\')[-1].split('_')[0])])
                self.label.append([self.charset.index(filename.split('\\')[-1].split('_')[0])])
            else:
                idx_list = []
                for item in list(filename.split('\\')[-1].split('_')[0]):
                    idx = self.charset.index(item)
                    idx_list.append(idx)
                self.label.append(idx_list)
                # self.label.append([self.charset.index(item) for item in list(filename.split('\\')[-1].split('_')[0])])

        pass

    def __getitem__(self, idx):
        fn = self.tensors[idx]
        if self.channel == 3:
            img = Image.open(fn).convert('RGB')
        elif self.channel == 1:
            img = Image.open(fn).convert('L')
        else:
            raise ChannelIsNotAllowed("Image Channel must be 3 or 1!")
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(self.label[idx])
        return img, label

    def __len__(self):
        return len(self.label)


class GetLoader(object):
    def __init__(self, project_name):
        self.project_name = project_name
        conf = Config(self.project_name)
        self.config = conf.load_config()
        self.resize = self.config['Model']['RESIZE']
        self.batch_size = {
            "train": self.config['Train']['BATCH_SIZE'],
            "test": self.config['Train']['TEST_BATCH_SIZE']
        }
        self.channel = self.config['Model']['ImageChannel']
        transform_list = []
        if self.resize != [-1, -1]:
            transform_list.append(torchvision.transforms.Resize(tuple(self.resize)))
        transform_list.append(torchvision.transforms.ToTensor())
        if self.channel == 3:
            transform_list.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        elif self.channel == 1:
            transform_list.append(torchvision.transforms.Normalize((0.5), (0.5)))
        else:
            raise ChannelIsNotAllowed("Image Channel must be 3 or 1!")

        self.transform = torchvision.transforms.Compose(transform_list)
        self.loaders = {}
        for mode in [RunMode.Train, RunMode.Test]:
            dataset = LoadDataset(self.project_name, self.transform, mode)

            is_shuffle = True if mode == RunMode.Train else False
            self.loaders[mode.value] = DataLoader(dataset=dataset, batch_size=self.batch_size[mode.value],
                                                  shuffle=is_shuffle,
                                                  num_workers=4, pin_memory=True, collate_fn=self.collate_to_sparse)

    def collate_to_sparse(self, batch):
        values = []
        images = []
        shapes = []
        for n, (img, seq) in enumerate(batch):
            if len(seq) == 0: continue
            values.extend(seq)
            images.append(img.numpy())
            shapes.append(len(seq))
        images = torch.from_numpy(np.array(images))

        # target = torch.sparse.LongTensor(i.t(), v).to_dense()
        return [images, torch.FloatTensor(values), torch.IntTensor(shapes)]

    def get_loaders(self):
        return self.loaders
