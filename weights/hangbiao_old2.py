import os
import sys
import pymysql
from pathlib import Path
from configuration import config
import torch
import torch.backends.cudnn as cudnn
from threading import Thread
from multiprocessing import Process
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadStreams
from utils.general import (check_file, check_img_size, check_requirements,
                           non_max_suppression)
from utils.torch_utils import select_device, time_sync
from utils.Match import *
from flask import Flask,request,jsonify

@torch.no_grad()
def detect(weights=ROOT / 'hangbiao.pt',  # model.pt path(s)
           source=0,
           start_time=None,
           last_vid=None,
           camera_id=None):
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
    device = select_device(device)  # ????????????
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)  # ????????????
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # ??????????????????????????????32???

    # Half
    # ?????????????????????????????????
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    # view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # ???????????????
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # ????????????????????????
    select_sql = "SELECT id,visibility_id,camera_para FROM baseline \
               WHERE camera_id = %s and visibility_id=%s" % (camera_id,last_vid)

    ALL_buoy = SELECT_Sql(host, username, password, db_name, select_sql)
    Video_frames=0
    success_num = 0
    ALL_vid=[]
    ALL_conf=[]
    End_python=False
    #
    end_frames=100

    for path, im, im0s, vid_cap, s in dataset:
        # ?????????????????????????????????
        if len(ALL_buoy)>=1:
            # ???????????????
            if Video_frames==0:
                P = ALL_buoy[0][2].strip("()").split(",")[0]
                T = ALL_buoy[0][2].strip("()").split(",")[1]
                Z = ALL_buoy[0][2].strip("()").split(",")[2]
                baseline_id=int(ALL_buoy[0][0])
                visibility_id=int(ALL_buoy[0][1])
                ALL_vid.append(visibility_id)
                print("??????PTZ",SetPtz(P=P, T=T, Z=Z))
                #??????????????????
                # ??????PTZ??????????????????????????????????????????
                now_PTZ = GetPtz()

            if abs(now_PTZ[0]-float(P))<=0.2 and abs(now_PTZ[1]-float(T))<=0.2 and abs(now_PTZ[2]-float(Z))<=0.2:
                print("????????????")
                begin = 1
                if Video_frames==0:
                    time.sleep(5)
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
            for i, det in enumerate(pred): # det????????????????????????????????????

                # ??????????????????????????????
                if begin==1:
                    Video_frames += 1
                    print("???????????????", Video_frames)
                    print("???????????????????????????", success_num)
                    print("???????????????vid???", visibility_id)
                    #?????????????????????result_frames?????????
                    if Video_frames<=end_frames:
                        # ?????????????????????????????????
                        if len(det) == 0:  # ??????????????????????????????
                            break
                        else:
                            print("???????????????????????????")
                            conf=round(det.tolist()[0][4],2)
                            ALL_conf.append(conf)
                            success_num+=1
                            break
                    else:
                        if success_num/end_frames>=0.2:
                            print("??????????????????????????????")
                            #???????????????
                            f_conf=mean(ALL_conf)
                            print(f_conf)
                            insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility,conf)
                                                                    VALUES (%d,'%s',%d,%d,1,%f)
                                                                    """ % (
                                camera_id, start_time, baseline_id,visibility_id,f_conf)
                            Insert_Sql(start_time, host, username, password, db_name, insert_sql)

                            #??????????????????
                            select_sql = "SELECT id,visibility_id,camera_para FROM baseline \
                                                                          WHERE camera_id = %s and visibility_id=%s" % (
                            camera_id, visibility_id+1)
                            ALL_buoy = SELECT_Sql(host, username, password, db_name, select_sql)
                            if len(ALL_buoy)>1:
                                ALL_conf=[]
                                print("????????????????????????")
                                Video_frames = 0
                                success_num = 0
                                break
                            else:
                                return "???????????????????????????????????????????????????"
                        else:
                            print("???????????????????????????")
                            ALL_conf = []
                            insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility)
                                                                                                VALUES (%d,'%s',%d,%d,0)
                                                                                                """ % (
                                camera_id, start_time, baseline_id,visibility_id)
                            Insert_Sql(start_time, host, username, password, db_name, insert_sql)
                            if len(ALL_buoy)>1:
                                ALL_buoy.pop(0)
                                print("??????????????????????????????")
                                Video_frames = 0
                                success_num = 0
                                break
                            else:
                                for i in ALL_vid:
                                    if (i == (visibility_id-1)):
                                        End_python=True
                                if End_python==False:
                                    select_sql = "SELECT id,visibility_id,camera_para FROM baseline \
                                                   WHERE camera_id = %s and visibility_id=%s" % (camera_id, visibility_id-1)
                                    ALL_buoy = SELECT_Sql(host, username, password, db_name, select_sql)
                                    if len(ALL_buoy) >=1:
                                        Video_frames = 0
                                        success_num = 0
                                        break
                                    else:
                                        return "???????????????????????????????????????????????????"
                                else:
                                    return "???????????????????????????????????????????????????"
                else:
                    break
        else:
            return "???????????????????????????????????????????????????"



app =Flask(__name__)

@app.route("/visibility",methods=["POST"])
def visibility():
    camera_id = request.args.get('camera_id')
    camera_id=int(camera_id)
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name
    # ??????????????????????????????
    select_sql = "select id from equipment"
    ALL_id = SELECT_Sql(host, username, password, db_name, select_sql)
    a = False
    for id in ALL_id:
        if id[0] == camera_id:
            a = True
    if a :
        # ??????camera_id??????URL
        select_sql = "SELECT video_url FROM equipment \
                           WHERE id = %s" % (camera_id)
        video_url = SELECT_Sql(host, username, password, db_name, select_sql)

        # ???????????????????????????
        select_sql = "SELECT bid FROM visibility_result WHERE camera_id =  %s and is_visibility=1 order by id desc limit 1" % (
            camera_id)
        last_bid = SELECT_Sql(host, username, password, db_name, select_sql)
        if len(last_bid) != 0:
            select_sql = "SELECT visibility_id FROM baseline WHERE id = %s" % (last_bid[0][0])
            last_vid = SELECT_Sql(host, username, password, db_name, select_sql)
            last_vid = last_vid[0][0]
        else:
            last_vid = 4
        # ??????????????????
        start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        # result=detect(source=video_url[0][0], start_time=start_time, last_vid=last_vid,camera_id=camera_id)
        t = Process(target=detect,args=(ROOT / 'hangbiao.pt',video_url[0][0], start_time, last_vid,camera_id))
        t.start()
        return "0"
    else:
        return "-1"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
