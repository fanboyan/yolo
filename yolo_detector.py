#!/bin/env/pytorch python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 15:57
# @Author  : YAN
# @FileName: yolo_detector.py
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

class Detector(object):

    def __init__(self):
        self.data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz = [640, 640]  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.device = ''


    def init_model(self,weights):
        self.weights = weights  # model.pt path(s)

        self.device = select_device(self.device)  # 设置设备
        model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)  # 加载模型
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # 模型的最大步长（默认32）
        # Half
        # 使用半精度，默认不使用
        self.half &= (self.pt or self.jit or self.onnx or self.engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            model.model.half() if self.half else model.model.float()
        self.model = model

    def init_source(self,source):
        self.source = str(source)
        # Dataloader
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # 读取视频流
        self.dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)


if __name__ == '__main__':
    ship = Detector()
    ship.init_model(ROOT / 'weights/ships.pt')
    source="https://open.ys7.com/v3/openlive/G18183870_1_2.m3u8?expire=1677567819&id=420234950509830144&t=c784a19375d066e715d175a1d68ad8146b88a84447e03c7bb751f02c43aceb0b&ev=100"

    ship.init_source(source)
    for path, im, im0s, vid_cap, s in ship.dataset:
        im = torch.from_numpy(im).to(ship.device)
        im = im.ship.half() if ship.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = ship.model(im, augment=ship.augment, visualize=ship.visualize)
        # NMS
        pred = non_max_suppression(pred, ship.conf_thres, ship.iou_thres, ship.classes, ship.agnostic_nms, max_det=ship.max_det)
        # Process predictions

        for i, det in enumerate(pred):  # per image
            p, im0, frame = path[i], im0s[i].copy(), ship.dataset.count
            s += f'{i}:'
            print(len(det))
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {ship.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(det)
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
            for *xyxy, conf, cls in reversed(det):
                # 获取视频检测框
                xyxy = torch.tensor(xyxy).view(-1).tolist()
                print(cls)