#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/6 10:49
# @Author  : YAN
# @FileName: Pre_ship_height.py
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
ROOT = FILE.parents[0]  # YOLOv5 root directory
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
def pre_ship_height(weights=ROOT / 'boat428.pt',  # model.pt path(s)
                    source=0,
                    start_time = None,
                    ship_distance=0,
                    camera_id = None,
                    ):
    data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    imgsz = [576, 704]  # inference size (height, width)
    conf_thres = 0.3  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes = 0  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    source = str(source)
    # save_img:bool 判断是否要保存图片
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
    old_time = 0
    old_im0 = 0
    s_time=time_sync()
    for path, im, im0s, vid_cap, s in dataset:

        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

        un_time=time_sync()
        if un_time-s_time>10000:
            LOGGER.info('自动关闭')
            return "自动关闭"
        if (un_time - old_time) > 1:
            old_time = un_time

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
                    LOGGER.info(f'连续两帧图片是否相同,{(im0 == old_im0).all()}')
                old_im0 = im0
                # Rescale boxes from img_size to im0 size

                # 判断视频是否检测出来船
                if len(det):  # 没有船的情况
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):
                        # 获取视频检测框
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        LOGGER.info(f'视频检测出来的船舶,{xyxy}')
                        # 利用预测框计算出距离
                        perHeight = (xyxy[3] - xyxy[1])  # 检测框的像素高度
                        perWidth = (xyxy[2] - xyxy[0])  # 检测框的像素高度
                        now_PTZ = GetPtz(camera_id)

                        # print(now_PTZ)
                        #预测高度
                        # distance = getDistance(24.48083, 118.07100, AIS['lat'] / 600000, AIS['lon'] / 600000)
                        distance = ship_distance
                        height1 = height_to_camera(perHeight=perHeight, fy=576, distance=distance,Z= now_PTZ["z"])
                        height2=height_to_camera1(perHeight=perHeight,perWidth=perWidth,ship_Width=31)
                        height=round(height1,2)
                        print("船实际距离", distance)
                        print("当前焦距", now_PTZ["z"])
                        print("实际船长", 43)
                        print("图像中船像素高度",perHeight)
                        print("图像中船像素宽度",perWidth)
                        print("焦距预测高度：",height)
                        print("船长预测高度：",height2)

                        im0 = plot_one_box(xyxy, im0, label=str(height)+"m", color=(0, 0, 255), line_thickness=3)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

app = Flask(__name__)

@app.route("/ship_Height", methods=["POST"])
def Ship_Height():
    camera_id = request.args.get('camera_id')
    # ship_distance = request.args.get('ship_distance')
    camera_id = int(camera_id)
    # ship_distance = int(ship_distance)
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    # 判断传入参数是否正确
    select_sql = "SELECT ship_height FROM equipment WHERE id = %s" % camera_id
    is_open = SELECT_Sql(host, username, password, db_name, select_sql)
    # if is_open[0][0] == 0:
    if True:
        # 查询camera_id对应URL
        select_sql = "SELECT video_url FROM equipment WHERE id = %s" % camera_id
        video_url = SELECT_Sql(host, username, password, db_name, select_sql)
        # 记录查询时间
        start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        ship_distance=1000
        # 开始执行
        t = Process(target=pre_ship_height, args=(ROOT / 'boat428.pt', video_url[0][0], start_time, ship_distance, camera_id))

        t.start()
        return "0"
    else:
        return "-1"


if __name__ == "__main__":

    app.run(host="127.0.0.1", port=config.port, debug=True)
