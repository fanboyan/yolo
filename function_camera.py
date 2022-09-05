#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 16:14
# @Author  : YAN
# @FileName: function_camera.py
# @Software: PyCharm
# @Email   : 812008450@qq.com

import datetime
import requests
import json
from numpy import *
from random import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymysql
import time

from configuration import config
from function_sql import SELECT_Sql
from utils.general import LOGGER

# setPTZ 卡达凯斯
def SetPtz(P, T, Z):
    url = "http://kdks.cniship.com:7080/ship-client/index/setPTZ?range=" + P + "&tilt=" + T + "&multiple=" + Z

    try:
        # get请求方式
        respones = requests.get(url=url)
    except Exception as e:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        LOGGER.info(f'{str(now_time)} : 设置PTZ失败\n{e}')
    data = respones.text
    # 将 JSON 对象转换为 Python 字典
    data = json.loads(data)

    ALL_AIS = data['message']
    return ALL_AIS

# 得到当前PTZ 卡达凯斯
def GetPtz(camera_id=None):
    if camera_id==2:
        url = "http://kdks.cniship.com:7080/ship-client/index/getPTZ"
    elif camera_id==1:
        url = "http://yuandang.cniship.com:7889/ship-client/index/getCameraPTZ?channel=1"
    try:
        # get请求方式
        respones = requests.get(url=url)
    except Exception as e:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        LOGGER.info(f'{str(now_time)} : 拿取PTZ失败\n{e}')
    # get请求方式
    data = respones.text
    # 将 JSON 对象转换为 Python 字典

    data = json.loads(data)

    PTZ = data['result']
    return PTZ
def select_url(camera_id):
    select_sql = "SELECT video_url FROM equipment WHERE id = %s" % camera_id
    video_url = SELECT_Sql(config.host, config.username, config.password, config.db_name, select_sql)
    return video_url