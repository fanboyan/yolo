#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 16:50
# @Author  : YAN
# @FileName: Pre_visibility.py
# @Software: PyCharm
# @Email   : 812008450@qq.com

import time
from numpy import mean
from camera.function_time import get_now_time, get_now_hour
from visivility.function_visibility import judge_night, select_all_buoy, judge_close, uodate_close, insert_history, \
    insert_result0, insert_result1, ultimately, rotate
from utils.general import LOGGER
from visivility.visibility_settings import Visibility_settings
import torch
from yolo_detector import Detector


@torch.no_grad()
def buoy_visibility(source=0,
           start_time=None,
           disatance_scale=None,
           camera_id=None,
           is_auto=None,
           kill_time=None,
            ROOT = None

):
    settings=Visibility_settings()
    night_time = get_now_hour()
    night,weights,end_frames=judge_night(night_time)
    weights=ROOT / weights
    LOGGER.info(f'使用{weights}')
    buoy_detect = Detector()
    buoy_detect.init_model(weights)
    buoy_detect.init_source(source)
    now_time = get_now_time()

    all_buoy = select_all_buoy(camera_id,disatance_scale)
    #
    Video_frames = 0
    success_num = 0
    ALL_did = []
    ALL_conf = []
    End_python = False
    End_countdown=0
    end_basline_id=0

    for path, im, im0s, vid_cap, s in buoy_detect.dataset:

        #判断是否需要关闭
        close=judge_close(camera_id)
        if close == settings.close:
            LOGGER.info('visibility=0,已关闭')
            return "visibility=0,已关闭"

        # 判断是否为最后一个航标
        if len(all_buoy) >= 1:
            # 转动摄像头
            if Video_frames == 0:

                P,T,Z,baseline_id,visibility_id,distance_id,ALL_did,now_PTZ = rotate(all_buoy, ALL_did, now_time, camera_id)
                if End_countdown > settings.end_countdown:
                    uodate_close(camera_id, start_time)
                    LOGGER.info('旋转超时，自动关闭')
                    return "旋转超时"
                End_countdown += 1
            if abs(now_PTZ[0] - float(P)) <= settings.angle_error_range and abs(now_PTZ[1] - float(T)) <= settings.angle_error_range and abs(
                    now_PTZ[2] - float(Z)) <= settings.angle_error_range:
                now_time = get_now_time()
                LOGGER.info(f'{str(now_time)} : 旋转成功')
                End_countdown = 0
                begin = 1
                if Video_frames == 0:
                    time.sleep(settings.sleep_time)
            else:
                begin = 0

            buoy_detect.inference(im)
            # Process predictions
            for i, det in enumerate(buoy_detect.pred):  # det：所有视频检测出来的信息

                now_time = get_now_time()
                # 判断是否进行识别检测
                if begin == 1:
                    Video_frames += 1
                    LOGGER.info(f'{str(now_time)} : 当前帧数{Video_frames}')
                    LOGGER.info(f'{str(now_time)} : 成功检测到航标帧数{success_num}')
                    LOGGER.info(f'{str(now_time)} : 正在第几个能见度的id{visibility_id}')
                    # 判断是否进超过result_frames帧识别
                    if Video_frames <= end_frames:
                        # 判断视频是否检测出来
                        if len(det) == 0:  # 没有检测到航标的情况
                            break
                        else:
                            LOGGER.info(f'{str(now_time)} : 该帧成功检测出航标')
                            conf = round(det.tolist()[0][4], 2)
                            ALL_conf.append(conf)
                            success_num += 1
                            break
                    else:
                        if success_num / end_frames >= settings.frame_error_ratio:
                            # 插入数据库
                            f_conf = mean(ALL_conf)
                            LOGGER.info(f'{str(now_time)} : 检测完成，并成功识别!!!概率为：{str(f_conf)}')
                            insert_result1(camera_id, start_time, baseline_id, visibility_id, f_conf)
                            end_basline_id=baseline_id
                            # 查询是否继续
                            all_buoy = select_all_buoy(camera_id, distance_id + 1)
                            if len(all_buoy) > 1:
                                ALL_conf = []
                                LOGGER.info(f'{str(now_time)} : 进行上级航标检测')
                                Video_frames = 0
                                success_num = 0
                                break
                            else:
                                uodate_close(camera_id, start_time)
                                LOGGER.info(f'{str(now_time)} : 当前航标已检测完毕，已自动关闭程序')
                                if end_basline_id != 0:
                                    insert_history(camera_id, start_time, end_basline_id, is_auto)
                                return True
                        else:
                            LOGGER.info(f'{str(now_time)} :检测完成，识别失败')
                            ALL_conf = []
                            insert_result0(camera_id, start_time, baseline_id, visibility_id)

                            if len(all_buoy)> 1:
                                all_buoy.pop(0)
                                LOGGER.info(f'{str(now_time)} :进行同级其他航标检测')
                                Video_frames = 0
                                success_num = 0
                                break
                            else:
                                for i in ALL_did:
                                    if i == (distance_id - 1):
                                        End_python = True
                                if not End_python:
                                    all_buoy = select_all_buoy(camera_id, distance_id - 1)
                                    if len(all_buoy) >= 1:
                                        Video_frames = 0
                                        success_num = 0
                                        break
                                    else:
                                        ultimately(camera_id, start_time, now_time, baseline_id, visibility_id, end_basline_id, is_auto)
                                        return "当前航标已检测完毕，已自动关闭程序"
                                else:
                                    ultimately(camera_id, start_time, now_time, baseline_id, visibility_id, end_basline_id, is_auto)
                                    return "当前航标已检测完毕，已自动关闭程序"
                else:
                    break
        else:
            ultimately(camera_id, start_time, now_time, baseline_id, visibility_id, end_basline_id, is_auto)
            return "当前航标已检测完毕，已自动关闭程序"