#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 17:17
# @Author  : YAN
# @FileName: detect-new.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
import os
import sys
from pathlib import Path
from visivility.Pre_visibility import buoy_visibility
from configuration import config
from multiprocessing import Process
from height.Pre_ship_height import pre_ship_height
from camera.function_camera import select_url
from height.function_height import check_parameters_height, update_switch_height
from camera.function_time import get_now_time
from visivility.function_visibility import check_parameters_visibility,update_switch_visibility
from flask import Flask,request
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

app = Flask(__name__)

@app.route("/visibility", methods=["POST"])
def visibility():

    camera_id = request.args.get('camera_id')
    camera_id = int(camera_id)
    is_auto = request.args.get('is_auto')
    is_auto = int(is_auto)
    # 判断传入参数是否正确
    is_open = check_parameters_visibility(camera_id)
    kill_time = is_open[0][1]
    if is_open[0][0] == 0:
        # 查询camera_id对应URL
        video_url = select_url(camera_id)
        distance = 1
        # 记录查询时间
        start_time = get_now_time()
        #visibility更新
        update_switch_visibility(start_time, camera_id)
        # 开始执行
        t = Process(target=buoy_visibility, args=(video_url[0][0], start_time, distance, camera_id,is_auto, kill_time,ROOT))
        t.start()
        return "0"
    else:
        return "-1"
@app.route("/ship_Height", methods=["POST"])
def ship_Height():

    camera_id = request.args.get('camera_id')
    camera_id = int(camera_id)
    # 判断传入参数是否正确
    is_open = check_parameters_height(camera_id)
    kill_time = is_open[0][1]
    if is_open[0][0] == 0:
        # 查询camera_id对应URL
        video_url = select_url(camera_id)
        # 记录查询时间
        start_time =get_now_time()
        update_switch_height(start_time,camera_id)
        # 开始执行
        t = Process(target=pre_ship_height, args=(video_url[0][0], kill_time, camera_id,ROOT))
        t.start()
        return "0"
    else:
        return "-1"

if __name__ == "__main__":

    app.run(host="127.0.0.1", port=config.port, debug=True)
