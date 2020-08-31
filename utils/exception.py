#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/23 2:11
@Author:   sml2h3
@File:     exception
@Software: PyCharm
"""


class DataDirNotFoundError(Exception):
    pass


class ChannelIsNotAllowed(Exception):
    pass


class CnnNotFoundError(Exception):
    pass


class RnnNotFoundError(Exception):
    pass


class OptimizerNotFoundError(Exception):
    pass

class PredictLabelLengthIsNotMatch(Exception):
    pass
