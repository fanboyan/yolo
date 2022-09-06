#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 17:18
# @Author  : YAN
# @FileName: function_height.py
# @Software: PyCharm
# @Email   : 812008450@qq.com

from configuration import config
from camera.function_sql import SELECT_Sql, Update_Sql, Insert_Sql


# 判断传入参数是否正确
def check_parameters_height(camera_id):

    select_sql = "SELECT ship_height,kill_time FROM equipment WHERE id = %s" % camera_id
    is_open = SELECT_Sql(config.host, config.username, config.password, config.db_name, select_sql)
    return is_open

def update_switch_height(start_time,camera_id):
    update_sql = "update equipment set ship_height=1 where id= %d" % camera_id
    Update_Sql(start_time,config.host, config.username, config.password, config.db_name, update_sql)

def judge_close(camera_id):
    # 判断是否需要关闭
    select_sql = "SELECT ship_height FROM equipment WHERE id = %s" % (camera_id)
    ship_height = SELECT_Sql(config.host, config.username, config.password, config.db_name, select_sql)
    return ship_height[0][0]

def uodate_close(camera_id,start_time):
    update_sql = "update equipment set ship_height=0 where id= %d" % camera_id
    Update_Sql(start_time,config.host, config.username, config.password, config.db_name, update_sql)

def insert_detect(camera_id,now_time,xyxy):
    insert_sql = """INSERT INTO height_detect(camera_id,post_time,min_u,max_u,min_v,max_v)
                                                                               VALUES (%d,'%s',%d,%d,%d,%d)
                                                                              """ % (
        camera_id, now_time, xyxy[0], xyxy[2], xyxy[1], xyxy[3])
    Insert_Sql(now_time,config.host, config.username, config.password, config.db_name,insert_sql)