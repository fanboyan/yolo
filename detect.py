#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 16:13
# @Author  : YAN
# @FileName: detect_height.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
import os
import sys
from pathlib import Path
from configuration import config
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import Process

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from utils.Match import *
from flask import Flask,request,jsonify


@torch.no_grad()
def detect(source=0,
           start_time=None,
           disatance_scale=None,
           camera_id=None,
           is_auto=None,
           kill_time=None):
    night_time = datetime.datetime.strftime(datetime.datetime.now(), '%H')
    if int(night_time) > 20 or 0 <= int(night_time) <= 5:
        weights = ROOT / 'buoy_night.pt'
        LOGGER.info('使用buoy_night.pt')
        night = True
    else:
        night = False
        weights = ROOT / 'buoy_day.pt'
        LOGGER.info('使用buoy_day.pt')
    now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    LOGGER.info(f'{str(now_time)} : 程序启动')
    data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz = [640, 640]  # inference size (height, width)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes = 0  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)  # 设置设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)  # 加载模型
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # 模型的最大步长（默认32）

    # Half
    # 使用半精度，默认不使用
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    # view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # 读取视频流
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # 创建保存图片文件夹
    period = datetime.datetime.now()
    timestr = period.strftime("%Y_%m_%d")
    savedir_path = './images/' + str(timestr)
    if not os.path.exists(savedir_path):
        os.makedirs(savedir_path)

    # 查询需要检测航标
    select_sql = "SELECT id,visibility_id,camera_para,distance_scale FROM baseline " \
                 "WHERE camera_id = %s and distance_scale = %s" % (camera_id, disatance_scale)

    all_body = SELECT_Sql(host, username, password, db_name, select_sql)
    Video_frames = 0
    success_num = 0
    ALL_did = []
    ALL_conf = []
    End_python = False

    # 进行检测的帧率
    if night==False:
        end_frames = 100
    else:
        end_frames = 200
    End_countdown=0
    end_basline_id=0

    for path, im, im0s, vid_cap, s in dataset:

        #判断是否需要关闭
        select_sql = "SELECT visibility FROM equipment WHERE id = %s" % (camera_id)
        visibility = SELECT_Sql(host, username, password, db_name, select_sql)
        if visibility[0][0] == 0:
            LOGGER.info('visibility=0,已关闭')
            return "visibility=0,已关闭"

        # 判断是否为最后一个航标
        if len(all_body) >= 1:
            # 转动摄像头
            if Video_frames == 0:
                P = all_body[0][2].strip("()").split(",")[0]
                T = all_body[0][2].strip("()").split(",")[1]
                Z = all_body[0][2].strip("()").split(",")[2]

                baseline_id = int(all_body[0][0])
                visibility_id = int(all_body[0][1])
                distance_id = int(all_body[0][3])
                ALL_did.append(distance_id)

                SetPtz(P=P, T=T, Z=Z)
                LOGGER.info(f'{str(now_time)} : 正在旋转PTZ')

                # 记录开始时间
                # 判断PTZ值，等待并确保摄像头转动成功
                now_PTZ = GetPtz(camera_id=camera_id)
                if End_countdown > 100:
                    update_sql = "update equipment set visibility=0 where id= %d" % camera_id
                    Update_Sql(start_time, host, username, password, db_name, update_sql)
                    LOGGER.info('旋转超时，自动关闭')
                    return "旋转超时"
                End_countdown += 1
            if abs(now_PTZ[0] - float(P)) <= 0.2 and abs(now_PTZ[1] - float(T)) <= 0.2 and abs(
                    now_PTZ[2] - float(Z)) <= 0.2:
                now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
                LOGGER.info(f'{str(now_time)} : 旋转成功')
                End_countdown = 0
                begin = 1
                if Video_frames == 0:
                    time.sleep(6)
            else:
                begin = 0

            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            pred = model(im, augment=augment, visualize=visualize)

            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # det：所有视频检测出来的信息
                p, im0, frame = path[i], im0s[i].copy(), dataset.count

                now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

                # 判断是否进行识别检测
                if begin == 1:
                    Video_frames += 1
                    #保存图片
                    # if Video_frames % 20 == 0 and night == True:
                    #     now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H-%M-%S')
                    #     Pd = int(round(time_sync() % 1, 2) * 100)
                    #     saveimages = f'{savedir_path}/{now_time}-{Pd}.jpg'
                    #     cv2.imwrite(saveimages, im0)
                    #     LOGGER.info(f'保存成功：{saveimages}')
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

                        if success_num / end_frames >= 0.2:
                            # 插入数据库
                            f_conf = mean(ALL_conf)
                            LOGGER.info(f'{str(now_time)} : 检测完成，并成功识别!!!概率为：{str(f_conf)}')


                            insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility,conf)
                                                                    VALUES (%d,'%s',%d,%d,1,%f)
                                                                    """ % (
                                camera_id, start_time, baseline_id, visibility_id, f_conf)
                            Insert_Sql(start_time, host, username, password, db_name, insert_sql)
                            end_basline_id=baseline_id

                            # 查询是否继续
                            select_sql = "SELECT id,visibility_id,camera_para,distance_scale FROM baseline \
                                                                          WHERE camera_id = %s and distance_scale=%s" % (
                                camera_id, distance_id + 1)
                            all_body = SELECT_Sql(host, username, password, db_name, select_sql)

                            if len(all_body) > 1:
                                ALL_conf = []
                                LOGGER.info(f'{str(now_time)} : 进行上级航标检测')
                                Video_frames = 0
                                success_num = 0
                                break
                            else:
                                update_sql = "update equipment set visibility=0 where id= %d" % camera_id
                                Update_Sql(start_time, host, username, password, db_name, update_sql)
                                LOGGER.info(f'{str(now_time)} : 当前航标已检测完毕，已自动关闭程序')
                                if end_basline_id != 0:
                                    insert_sql = """INSERT INTO visibility_history(camera_id,set_time,baseline_id,is_auto)
                                                                                                            VALUES (%d,'%s',%d,%d)
                                                                                                             """ % (
                                        camera_id, start_time, end_basline_id, is_auto)
                                    Insert_Sql(start_time, host, username, password, db_name, insert_sql)
                                return True
                        else:
                            LOGGER.info(f'{str(now_time)} :检测完成，识别失败')
                            ALL_conf = []
                            insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility)
                                                                                                VALUES (%d,'%s',%d,%d,0)
                                                                                                """ % (
                                camera_id, start_time, baseline_id, visibility_id)
                            Insert_Sql(start_time, host, username, password, db_name, insert_sql)
                            if len(all_body)> 1:
                                all_body.pop(0)
                                LOGGER.info(f'{str(now_time)} :进行同级其他航标检测')
                                Video_frames = 0
                                success_num = 0
                                break
                            else:
                                for i in ALL_did:
                                    if i == (distance_id - 1):
                                        End_python = True
                                if not End_python:
                                    select_sql = "SELECT id,visibility_id,camera_para,distance_scale FROM baseline \
                                                                          WHERE camera_id = %s and distance_scale=%s" % (
                                        camera_id, distance_id - 1)
                                    all_body = SELECT_Sql(host, username, password, db_name, select_sql)
                                    if len(all_body) >= 1:
                                        Video_frames = 0
                                        success_num = 0
                                        break
                                    else:
                                        update_sql = "update equipment set visibility=0 where id= %d" % (camera_id)
                                        Update_Sql(start_time, host, username, password, db_name, update_sql)
                                        LOGGER.info(f'{str(now_time)} : 当前航标已检测完毕，已自动关闭程序')

                                        Remain_buy_insert(camera_id, start_time, baseline_id, visibility_id, host,
                                                          username,
                                                          password, db_name)
                                        if end_basline_id != 0:
                                            insert_sql = """INSERT INTO visibility_history(camera_id,set_time,baseline_id,is_auto)
                                                                                                                    VALUES (%d,'%s',%d,%d)
                                                                                                                     """ % (
                                                camera_id, start_time, end_basline_id, is_auto)
                                            Insert_Sql(start_time, host, username, password, db_name, insert_sql)
                                        return True
                                else:
                                    update_sql = "update equipment set visibility=0 where id= %d" % (camera_id)
                                    Update_Sql(start_time, host, username, password, db_name, update_sql)
                                    LOGGER.info(f'{str(now_time)} : 当前航标已检测完毕，已自动关闭程序')
                                    Remain_buy_insert(camera_id, start_time, baseline_id, visibility_id, host, username,
                                                      password, db_name)
                                    if end_basline_id != 0:
                                        insert_sql = """INSERT INTO visibility_history(camera_id,set_time,baseline_id,is_auto)
                                                                                                                VALUES (%d,'%s',%d,%d)
                                                                                                                 """ % (
                                            camera_id, start_time, end_basline_id, is_auto)
                                        Insert_Sql(start_time, host, username, password, db_name, insert_sql)
                                    return True
                else:
                    break
        else:
            update_sql = "update equipment set visibility=0 where id= %d" % (camera_id)
            Update_Sql(start_time, host, username, password, db_name, update_sql)
            Remain_buy_insert(camera_id, start_time, baseline_id, visibility_id, host, username,
                              password, db_name)
            if end_basline_id!=0:
                insert_sql = """INSERT INTO visibility_history(camera_id,set_time,baseline_id,is_auto)
                                                                                        VALUES (%d,'%s',%d,%d)
                                                                                         """ % (
                    camera_id, start_time, end_basline_id,is_auto)
                Insert_Sql(start_time, host, username, password, db_name, insert_sql)
            LOGGER.info('检测完成自动关闭')
            return "当前航标已检测完毕，已自动关闭程序"
@torch.no_grad()
def pre_ship_height(
        source=0,
        kill_time=None,
        camera_id=None,
        device='',
        imgsz=None,
        stride=None,
        pt=None,
        model=None,
):
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz = [576, 704]  # inference size (height, width)
    conf_thres = 0.3  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = True  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    dnn = False  # use OpenCV DNN for ONNX inference
    weights = 'boat8-12.pt'  # model.pt path(s)
    device = ''
    half = True  # use FP16 half-precision inference

    # Load model
    device = select_device(device)  # 设置设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)  # 加载模型
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # 模型的最大步长（默认32）
    # Half
    # 使用半精度，默认不使用
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    t1 = time_sync()
    source = str(source)
    # Dataloader
    # view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # 读取视频流
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    # bs = len(dataset)  # batch_size
    #
    # # Run inference
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    t2 = time_sync()
    print(t2 - t1)
    old_time = 0
    old_im0 = 0
    s_time = time_sync()
    for path, im, im0s, vid_cap, s in dataset:

        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

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

            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = model(im, augment=augment, visualize=visualize)
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # Process predictions

            for i, det in enumerate(pred):  # per image

                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}:'

                if type(old_im0) != type(0):
                    if (im0 == old_im0).all():
                        LOGGER.info(f'连续两帧图片是相同,{(im0 == old_im0).all()}')
                old_im0 = im0
                # Rescale boxes from img_size to im0 size

                # 判断视频是否检测出来船
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
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

                #         im0 = plot_one_box(xyxy, im0, label="ship", color=(0, 0, 255), line_thickness=3)
                #
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond

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
        start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        # 记录查询时间

        update_sql = "update equipment set visibility=1 where id= %d" % camera_id
        Update_Sql(start_time, host, username, password, db_name, update_sql)

        # 开始执行
        t = Process(target=detect, args=(video_url[0][0], start_time, distance, camera_id,is_auto, kill_time))
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
