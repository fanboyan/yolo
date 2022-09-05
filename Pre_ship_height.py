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
from function_height import judge_close, uodate_close, insert_detect
from function_time import get_now_time
from height_settings import Height_settings
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
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
    settings = Height_settings()
    height_detect = Detector()
    height_detect.init_model(ROOT / settings.weights)
    LOGGER.info(f'使用{settings.weights}')
    height_detect.init_source(source)
    old_time = 0
    s_time = time_sync()
    for path, im, im0s, vid_cap, s in height_detect.dataset:
        #判断是否需要关闭
        ship_height = judge_close(camera_id)
        if ship_height == settings.close:
            LOGGER.info('ship_height=0,已关闭')
            return "ship_height:0,已关闭"
        height_detect.inference(im)
        now_time = get_now_time()
        un_time = time_sync()
        if un_time - s_time > (kill_time * 600):
            uodate_close(camera_id, now_time)
            LOGGER.info('自动关闭')
            return "自动关闭"
        if (un_time - old_time) > 0:
            old_time = un_time
            LOGGER.info(f'当前运行时间：{un_time - s_time}')
            for i, det in enumerate(height_detect.pred):  # per image
                height_detect.pre_image(path, im0s, s, i, det)
                # 判断视频是否检测出来船
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        # 获取视频检测框
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        LOGGER.info(f'视频检测出来的船舶,{xyxy}')
                        insert_detect(camera_id,now_time,xyxy)
                        # im0 = plot_one_box(xyxy, height_detect.im0, label="ship", color=(0, 0, 255), line_thickness=3)
                # height_detect.video_Show()