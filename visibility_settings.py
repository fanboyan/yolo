#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 16:35
# @Author  : YAN
# @FileName: visibility_settings.py
# @Software: PyCharm
# @Email   : 812008450@qq.com

class Visibility_settings(object):
    "yolo detect"
    def __init__(self):
        #
        self.close=0
        self.end_countdown=100
        self.angle_error_range=0.2
        self.sleep_time=6
        self.frame_error_ratio=0.2