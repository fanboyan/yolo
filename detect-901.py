#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 17:17
# @Author  : YAN
# @FileName: detect-901.py
# @Software: PyCharm
# @Email   : 812008450@qq.com


import os
import sys
from pathlib import Path

from Pre_visibility import buoy_visibility
from configuration import config
import torch
from multiprocessing import Process
from Pre_ship_height import pre_ship_height
from function_time import get_now_time
from yolo_detector import Detector

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.Match import *
from flask import Flask,request,jsonify



app = Flask(__name__)

@app.route("/visibility", methods=["POST"])
def visibility():
    camera_id = request.args.get('camera_id')
    camera_id = int(camera_id)
    is_auto = request.args.get('is_auto')
    is_auto = int(is_auto)
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    # 判断传入参数是否正确
    select_sql = "SELECT visibility,kill_time FROM equipment WHERE id = %s" % camera_id
    is_open = SELECT_Sql(host, username, password, db_name, select_sql)
    kill_time = is_open[0][1]
    if is_open[0][0] == 0:
        # 查询camera_id对应URL
        select_sql = "SELECT video_url FROM equipment WHERE id = %s" % camera_id
        video_url = SELECT_Sql(host, username, password, db_name, select_sql)

        distance = 1
        # 记录查询时间
        start_time = get_now_time()
        # 记录查询时间

        update_sql = "update equipment set visibility=1 where id= %d" % camera_id
        Update_Sql(start_time, host, username, password, db_name, update_sql)

        # 开始执行
        t = Process(target=buoy_visibility, args=(video_url[0][0], start_time, distance, camera_id,is_auto, kill_time))
        t.start()
        return "0"
    else:
        return "-1"
@app.route("/ship_Height", methods=["POST"])
def Ship_Height():
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    camera_id = request.args.get('camera_id')
    camera_id = int(camera_id)
    # 判断传入参数是否正确
    select_sql = "SELECT ship_height,kill_time FROM equipment WHERE id = %s" % camera_id
    is_open = SELECT_Sql(host, username, password, db_name, select_sql)
    kill_time = is_open[0][1]
    if is_open[0][0] == 0:
        # 查询camera_id对应URL
        select_sql = "SELECT video_url FROM equipment WHERE id = %s" % camera_id
        video_url = SELECT_Sql(host, username, password, db_name, select_sql)
        # 记录查询时间
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

        update_sql = "update equipment set ship_height=1 where id= %d" % camera_id
        Update_Sql(now_time, host, username, password, db_name, update_sql)

        # 开始执行
        t = Process(target=pre_ship_height,
                    args=(video_url[0][0], kill_time, camera_id))
        t.start()
        # pre_ship_height(video_url[0][0],kill_time,camera_id,device,imgsz,stride,pt,model)
        return "0"
    else:
        return "-1"

if __name__ == "__main__":


    app.run(host="127.0.0.1", port=config.port, debug=True)
