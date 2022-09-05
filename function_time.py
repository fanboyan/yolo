#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 15:54
# @Author  : YAN
# @FileName: function_time.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
import datetime

def get_now_time():

    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def get_now_hour():

    return datetime.datetime.strftime(datetime.datetime.now(), '%H')