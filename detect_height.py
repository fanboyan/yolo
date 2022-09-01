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
def pre_ship_height(
                    source=0,
                    kill_time = None,
                    camera_id = None,
                    device='',
                    imgsz=None,
                    stride=None,
                    pt=None,
                    model=None,
                    ):
    t1=time_sync()
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
    print(t2-t1)
    old_time = 0
    old_im0 = 0
    s_time=time_sync()
    for path, im, im0s, vid_cap, s in dataset:

        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

        un_time=time_sync()
        if un_time-s_time>(kill_time*60):
            update_sql = "update equipment set ship_height=0 where id= %d" % camera_id
            Update_Sql(now_time, host, username, password, db_name, update_sql)
            LOGGER.info('自动关闭')
            return "自动关闭"


        if (un_time - old_time) > 1:
            old_time = un_time

            select_sql = "SELECT ship_height FROM equipment WHERE id = %s" % (camera_id)
            ship_height = SELECT_Sql(host, username, password, db_name, select_sql)

            if ship_height[0][0] == 0:
                LOGGER.info('已关闭')
                return "已关闭"

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
                        LOGGER.info(f'视频检测出来的船舶,{xyxy}')
                        insert_sql = """INSERT INTO height_detect(camera_id,post_time,min_u,max_u,min_v,max_v)
                                                                           VALUES (%d,'%s',%d,%d,%d,%d)
                                                                          """ % (
                            camera_id, now_time, xyxy[0], xyxy[2], xyxy[1], xyxy[3])
                        Insert_Sql(now_time, host, username, password, db_name, insert_sql)

                        im0 = plot_one_box(xyxy, im0, label="ship", color=(0, 0, 255), line_thickness=3)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
app = Flask(__name__)

@app.route("/ship_Height", methods=["POST"])
def Ship_Height():
    camera_id = request.args.get('camera_id')
    camera_id = int(camera_id)
    # 判断传入参数是否正确
    select_sql = "SELECT ship_height,kill_time FROM equipment WHERE id = %s" % camera_id
    is_open = SELECT_Sql(host, username, password, db_name, select_sql)
    kill_time=is_open[0][1]
    if True:
        # 查询camera_id对应URL
        select_sql = "SELECT video_url FROM equipment WHERE id = %s" % camera_id
        video_url = SELECT_Sql(host, username, password, db_name, select_sql)
        # 记录查询时间
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

        update_sql = "update equipment set ship_height=1 where id= %d" % camera_id
        Update_Sql(now_time, host, username, password, db_name, update_sql)


        # 开始执行
        t = Process(target=pre_ship_height, args=(video_url[0][0],kill_time,camera_id,device,imgsz,stride,pt,model))

        t.start()

        # pre_ship_height(video_url[0][0],kill_time,camera_id,device,imgsz,stride,pt,model)
        return "0"
    else:
        return "-1"


if __name__ == "__main__":
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz = [576, 704]   # inference size (height, width)
    conf_thres = 0.3  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = 0  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    dnn = False  # use OpenCV DNN for ONNX inference
    weights ='boat428.pt'  # model.pt path(s)
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

    app.run(host="127.0.0.1", port=config.port, debug=True)
