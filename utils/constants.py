#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/23 0:22
@Author:   sml2h3
@File:     constants
@Software: PyCharm
"""
from enum import Enum, unique
import json


@unique
class DataType(Enum):
    ClassFication = "classfication"
    Detection = "detection"


@unique
class Labels(Enum):
    Numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    Alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z"]
    NumbersAndAlphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i",
                          "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


@unique
class RunMode(Enum):
    Train = "train"
    Test = "test"
    Predict = "predict"


@unique
class CNN(Enum):
    MobileNetV2 = "MobileNetV2"
    EfficientNetb0 = "EfficientNet-b0"


@unique
class RNN(Enum):
    LSTM = "LSTM"


@unique
class OPTIMIZER(Enum):
    Momentum = "Momentum"
    Adam = "Adam"

