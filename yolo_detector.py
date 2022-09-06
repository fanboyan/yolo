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
import cv2
import torch
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import ( check_img_size,  non_max_suppression,scale_coords)
from utils.torch_utils import select_device

class Detector(object):
    "yolo detect"
    def __init__(self):
        #
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

    def inference(self,im):
        self.im = torch.from_numpy(im).to(self.device)
        self.im = self.im.ship.half() if self.half else self.im.float()  # uint8 to fp16/32
        self.im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(self.im.shape) == 3:
            self.im = self.im[None]  # expand for batch dim
        # Inference
        pred = self.model(self.im, augment=self.augment, visualize=self.visualize)
        # NMS
        self.pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
    def pre_image(self,path, im0s, s ,i, det ):
        self.p, self.im0, self.frame = path[i], im0s[i].copy(), self.dataset.count
        s += f'{i}:'

        det[:, :4] = scale_coords(self.im.shape[2:], det[:, :4], self.im0.shape).round()
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
    def  video_Show(self):

        cv2.imshow(str(self.p), self.im0)
        cv2.waitKey(1)  # 1 millisecond