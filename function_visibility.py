#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 17:45
# @Author  : YAN
# @FileName: function_visibility.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
from function_sql import SELECT_Sql, Update_Sql, Insert_Sql
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

def judge_close(camera_id):
    select_sql = "SELECT visibility FROM equipment WHERE id = %s" % (camera_id)
    visibility = SELECT_Sql(config.host, config.username, config.password, config.db_name, select_sql)

    return visibility[0][0]

def uodate_close(camera_id,start_time):
    update_sql = "update equipment set visibility=0 where id= %d" % camera_id
    Update_Sql(start_time,config.host, config.username, config.password, config.db_name, update_sql)

def insert_result0(camera_id, start_time, baseline_id, visibility_id):
    insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility,conf)
                                            VALUES (%d,'%s',%d,%d,0)
                                            """ % (
        camera_id, start_time, baseline_id, visibility_id)
    Insert_Sql(start_time,config.host, config.username, config.password, config.db_name,  insert_sql)

def insert_result1(camera_id, start_time, baseline_id, visibility_id, f_conf):
    insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility,conf)
                                             VALUES (%d,'%s',%d,%d,1,%f)
                                             """ % (
        camera_id, start_time, baseline_id, visibility_id, f_conf)
    Insert_Sql(start_time, config.host, config.username, config.password, config.db_name, insert_sql)

def insert_history(camera_id, start_time, end_basline_id, is_auto):
    insert_sql = """INSERT INTO visibility_history(camera_id,set_time,baseline_id,is_auto)
                                                                                                               VALUES (%d,'%s',%d,%d)
                                                                                                                """ % (
        camera_id, start_time, end_basline_id, is_auto)
    Insert_Sql(start_time,config.host, config.username, config.password, config.db_name,  insert_sql)