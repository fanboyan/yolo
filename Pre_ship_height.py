#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 15:39
# @Author  : YAN
# @FileName: Pre_ship_height.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from configuration import config
import torch
from yolo_detector import Detector
from utils.torch_utils import time_sync
from utils.Match import *



@torch.no_grad()
def pre_ship_height(
        source=None,
        kill_time=None,
        camera_id=None,
):
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name

    height_detect = Detector()
    height_detect.init_model(ROOT / 'weights/buoy_day.pt')
    height_detect.init_source(source)
    old_time = 0

    s_time = time_sync()
    for path, im, im0s, vid_cap, s in height_detect.dataset:

        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

        height_detect.inference(im)

        un_time = time_sync()
        if un_time - s_time > (kill_time * 600):
            update_sql = "update equipment set ship_height=0 where id= %d" % camera_id
            Update_Sql(now_time, host, username, password, db_name, update_sql)
            LOGGER.info('超时,自动关闭')
            return "自动关闭"

        if (un_time - old_time) > 0:
            old_time = un_time

            select_sql = "SELECT ship_height FROM equipment WHERE id = %s" % (camera_id)
            ship_height = SELECT_Sql(host, username, password, db_name, select_sql)

            if ship_height[0][0] == 0:
                LOGGER.info('ship_height=0,已关闭')
                return "ship_height:0,已关闭"

            LOGGER.info(f'当前运行时间：{un_time - s_time}')

            for i, det in enumerate(height_detect.pred):  # per image

                height_detect.pre_image(path, im0s, s, i, det)

                # 判断视频是否检测出来船
                if len(det):

                    for *xyxy, conf, cls in reversed(det):
                        # 获取视频检测框
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        print(cls)
                        LOGGER.info(f'视频检测出来的船舶,{xyxy}')
                        insert_sql = """INSERT INTO height_detect(camera_id,post_time,min_u,max_u,min_v,max_v)
                                                                           VALUES (%d,'%s',%d,%d,%d,%d)
                                                                          """ % (
                            camera_id, now_time, xyxy[0], xyxy[2], xyxy[1], xyxy[3])
                        Insert_Sql(now_time, host, username, password, db_name, insert_sql)

                        # im0 = plot_one_box(xyxy, height_detect.im0, label="ship", color=(0, 0, 255), line_thickness=3)

                # height_detect.video_Show()