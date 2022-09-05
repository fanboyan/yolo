#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 17:45
# @Author  : YAN
# @FileName: function_visibility.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
from function_camera import SetPtz, GetPtz
from function_sql import SELECT_Sql, Update_Sql, Insert_Sql
from configuration import config
from utils.general import LOGGER


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

#循环插入剩余
def Remain_buy_insert(camera_id,start_time,baseline_id,visibility_id):
    for i in range(30 - baseline_id):
        insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility)
                                                                                                        VALUES (%d,'%s',%d,%d,0)
                                                                                                        """ % (
            camera_id, start_time, baseline_id+i+1, visibility_id)
        Insert_Sql(start_time,config.host, config.username, config.password, config.db_name,  insert_sql)
def ultimately(camera_id, start_time,now_time, baseline_id, visibility_id,end_basline_id, is_auto):
    uodate_close(camera_id, start_time)
    LOGGER.info(f'{str(now_time)} : 当前航标已检测完毕，已自动关闭程序')
    Remain_buy_insert(camera_id, start_time, baseline_id, visibility_id)
    if end_basline_id != 0:
        insert_history(camera_id, start_time, end_basline_id, is_auto)

def rotate(all_buoy,ALL_did,now_time,camera_id):
    P = all_buoy[0][2].strip("()").split(",")[0]
    T = all_buoy[0][2].strip("()").split(",")[1]
    Z = all_buoy[0][2].strip("()").split(",")[2]

    baseline_id = int(all_buoy[0][0])
    visibility_id = int(all_buoy[0][1])
    distance_id = int(all_buoy[0][3])
    ALL_did.append(distance_id)

    SetPtz(P=P, T=T, Z=Z)
    LOGGER.info(f'{str(now_time)} : 正在旋转PTZ')
    # 记录开始时间
    # 判断PTZ值，等待并确保摄像头转动成功
    now_PTZ = GetPtz(camera_id=camera_id)
    return P,T,Z,baseline_id,visibility_id,distance_id,ALL_did,now_PTZ

 # 判断传入参数是否正确
def check_parameters_visibility(camera_id):
    select_sql = "SELECT visibility,kill_time FROM equipment WHERE id = %s" % camera_id
    is_open = SELECT_Sql(config.host, config.username, config.password, config.db_name, select_sql)
    return is_open

def update_switch_visibility(start_time,camera_id):
    update_sql = "update equipment set visibility=1 where id= %d" % camera_id
    Update_Sql(start_time,config.host, config.username, config.password, config.db_name, update_sql)