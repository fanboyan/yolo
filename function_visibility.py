#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 17:45
# @Author  : YAN
# @FileName: function_visibility.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
from function_sql import SELECT_Sql
from configuration import config

def judge_night(night_time):
    if int(night_time) > 20 or 0 <= int(night_time) <= 5:
        weights = 'weights/buoy_night.pt'
        night = True
        end_frames = 200
    else:
        night = False
        weights = 'weights/buoy_day.pt'
        end_frames = 100
    return night,weights,end_frames

def select_all_buoy(camera_id,disatance_scale):
    select_sql = "SELECT id,visibility_id,camera_para,distance_scale FROM baseline " \
                 "WHERE camera_id = %s and distance_scale = %s" % (camera_id, disatance_scale)

    all_buoy = SELECT_Sql(config.host, config.username, config.password, config.db_name, select_sql)
    return all_buoy