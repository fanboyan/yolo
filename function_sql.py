#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 15:59
# @Author  : YAN
# @FileName: function_sql.py
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
from utils.general import LOGGER

# 插入数据库
def Insert_Sql(now_time, host, username, password, db_name, insert_sql):
    connect = pymysql.connect(host=host, user=username, password=password, db=db_name, charset='utf8')

    # 获取游标对象
    cursor = connect.cursor()
    try:
        # 插入数据
        cursor.execute(insert_sql)
        connect.commit()
        LOGGER.info(f'{now_time},插入成功')
    except Exception as e:
        connect.rollback()
        LOGGER.info(f'{now_time},插入失败, {e}')
    cursor.close()
    connect.close()


# 修改数据库数据
def Update_Sql(now_time, host, username, password, db_name, update_sql):
    connect = pymysql.connect(host=host, user=username, password=password, db=db_name, charset='utf8')

    # 获取游标对象
    cursor = connect.cursor()

    # # 将时间戳转为时间格式
    # dateTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    try:
        # 修改数据
        cursor.execute(update_sql)
        connect.commit()
        LOGGER.info(f'{now_time},修改成功')
    except Exception as e:
        connect.rollback()
        LOGGER.info(f'{now_time},修改失败, {e}')
    cursor.close()
    connect.close()


# 查询数据库
def SELECT_Sql(host, username, password, db_name, select_sql):
    connect = pymysql.connect(host=host, user=username, password=password, db=db_name, charset='utf8')

    # 获取游标对象
    cursor = connect.cursor()
    # SQL 查询语句
    try:
        # 执行SQL语句
        cursor.execute(select_sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        results = list(results)
    except Exception as e:
        connect.rollback()
        print("查询失败",e)
    cursor.close()
    connect.close()
    return results

#循环插入剩余
def Remain_buy_insert(camera_id,start_time,baseline_id,visibility_id,host,username,password,db_name):
    for i in range(30 - baseline_id):
        insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility)
                                                                                                        VALUES (%d,'%s',%d,%d,0)
                                                                                                        """ % (
            camera_id, start_time, baseline_id+i+1, visibility_id)
        Insert_Sql(start_time, host, username, password, db_name, insert_sql)
    return 1