import os
import sys
from pathlib import Path
from configuration import config
import torch
import torch.backends.cudnn as cudnn

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
import math


class Point:
    """
    2D坐标点
    """
    def __init__(self, x, y):
        self.X = x
        self.Y = y


class Line:
    def __init__(self, point1, point2):
        """
        初始化包含两个端点
        :param point1:
        :param point2:
        """
        self.Point1 = point1
        self.Point2 = point2


def normalized(data,data_max,data_min):
    result = (data - data_min)/(data_max-data_min)
    return result




@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=0,
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=[576, 704],  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    host = config.host
    username = config.username
    password = config.password
    db_name = config.db_name

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
    dt, seen = [0.0, 0.0, 0.0], 0
    old_time = 0
    old_im0 = 0
    for path, im, im0s, vid_cap, s in dataset:

        t1 = time_sync()
        AISData = getAISData(port=config.port)
        PTZ = GainPtz(port=config.port)
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
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}:'
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % im.shape[2:]  # print string

            # 间隔1秒取一次
            now_time = GainTime(port=config.port)

            # un_time1 = int(time.mktime(now_time.timetuple()))  # dateTime格式转时间戳

            un_time1 = int(time.mktime(time.strptime(now_time, "%Y-%m-%d %H:%M:%S")))

            # 判断是否识别到船舶，并间隔大于等于一秒
            if (un_time1 - old_time) >= 1:
                ship_number = len(det)  # 视频检测出的船舶数量

                old_time = un_time1  # 用来判断间隔时间

                #判断两帧图片是否相同
                if type(old_im0) != type(0):
                    LOGGER.info(f'连续两帧图片是否相同,{(im0 == old_im0).all()}')
                old_im0 = im0

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 判断视频是否检测出来船
                if ship_number == 0:  # 没有船的情况
                    LOGGER.info('视频没有检测出来')
                    insert_sql = """INSERT INTO camera_result(reset_time,ship_number)
                                                       VALUES ('%s',0)
                                                                                   """ % now_time

                    Insert_Sql(now_time, host, username, password, db_name, insert_sql)
                else:

                    # 保存预测结果
                    ALL_res = []  # 建立空list存储所有匹配结果
                    ALL_xyxy = []  # 建立空list存储视屏框
                    # det：所有视频检测出来的信息
                    # print(reversed(det))

                    for *xyxy, conf, cls in reversed(det):

                        # 获取视频检测框
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        LOGGER.info(f'视频检测出来的船舶,{xyxy}')
                        m_res = []  # 视频一个框与所有AIS数据的匹配结果

                        # 利用预测框计算出距离
                        perHeight = (xyxy[3] - xyxy[1])  # 检测框的像素高度

                        distance_xyxy = distance_to_camera(perWidth=perHeight, Z=PTZ["z"], knownWidth=20)
                        # 计算相机预测距离
                        # print("预测框计算出距离",distance_xyxy)

                        middle_u = 352
                        middle_v = 576
                        distance_max = math.sqrt((0-middle_u)**2+(0-middle_v)**2)
                        distance_min = 0

                        for AIS in AISData:
                            # 计算方位角

                            # AIS转换坐标中心点
                            # AIS['minU'] = 704-AIS['minU']
                            # AIS['maxU'] = 704-AIS['maxU']
                            # AIS['minV'] = 576-AIS['minV']
                            # AIS['maxV'] = 576-AIS['maxV']

                            AIS_u = AIS['minU'] + (AIS["maxU"] - AIS['minU']) / 2
                            AIS_v = AIS['minV'] + (AIS["maxV"] - AIS['minV']) / 2
                            # print(AIS_v,AIS_v)

                            # 视频检测框中心点
                            xyxy_u = xyxy[0] + (xyxy[2] - xyxy[0]) / 2
                            xyxy_v = xyxy[1] + (xyxy[3] - xyxy[1]) / 2

                            # 两个中心点与点(352, 576)连线

                            L1 = Line(Point(middle_u, middle_v), Point(AIS_u, AIS_v))
                            L2 = Line(Point(middle_u, middle_v), Point(xyxy_u, xyxy_v))

                            angle = normalized(GetAngle(L1, L2), 180, 0)

                            distance_ais = math.sqrt((AIS_u-middle_u)**2+(AIS_v-middle_v)**2)
                            distance_xyxy = math.sqrt((xyxy_u-middle_u)**2+(xyxy_v-middle_v)**2)
                            distance = normalized(math.fabs(distance_ais-distance_xyxy), distance_max, distance_min)

                            # print(AIS['mmsi'], "方位角：", res)

                            # distance = getDistance(24.48083, 118.07100, AIS['lat'] / 600000, AIS['lon'] / 600000)

                            # d_tmp = abs(distance - distance_xyxy)

                            # print(AIS["mmsi"], "船实际距离", distance)
                            # print("预测框计算出距离", distance_xyxy)
                            # print("船舶mmsi", AIS["mmsi"])
                            # print("AIS转换后中心点0", AIS_u, AIS_v)
                            # print("图像识别中心点0", xyxy_u, xyxy_v)
                            # print("两点之间的方位角",res)
                            # print("处理后的方位角",res+D/100)

                            # m_res.append(res+ d_tmp / 100)

                            res = 0.5 * angle + 0.5 * distance
                            # print("AIS:", AIS['mmsi'], AIS['name'], AIS['maxU'], AIS['maxV'], AIS['minU'], AIS['minV'])
                            # print("xyxy:", xyxy)
                            # print("结果为",res)
                            m_res.append(res)

                        ALL_res.append(m_res)  # 往总的结果里加入
                        ALL_xyxy.append(xyxy)  # 存视频检测结果


                    if len(AISData) > 0:
                        # 循环遍历ALL_xyxy
                        m_xyxy = ALL_xyxy  # 用来弹出匹配成功的xyxy
                        for xyxy in ALL_xyxy:
                            # 筛选出len(c)组方位角中的最小值进行匹配，并弹出最小值所在list与匹配到的AIS信息
                            mm = 0.2  # 限值范围
                            count = 0
                            # 遍历ALL_res
                            for m in range(len(ALL_res)):
                                # 筛选出方位角中的最小值
                                if not ALL_res[m]:
                                    AIS_mmsi = "NO MATCH"
                                else:
                                    if float(min(ALL_res[m])) < mm:
                                        mm = min(ALL_res[m])
                                        AIS_index = ALL_res[m].index(min(ALL_res[m]))
                                        # AISData[AIS_index]['minU'] = 704 - AIS['minU']
                                        # AISData[AIS_index]['maxU'] = 704 - AIS['maxU']
                                        # AISData[AIS_index]['minV'] = 576 - AIS['minV']
                                        # AISData[AIS_index]['maxV'] = 576 - AIS['maxV']
                                        AIS_mmsi = str(AISData[AIS_index]["name"])
                                        sqlAIS = AISData[AIS_index]
                                        # print(sqlAIS)
                                        b = m
                                        count = 1
                                # 判断是否匹配成功
                            if count == 0:
                                # print("未匹配上")
                                AIS_mmsi = "NO MATCH"
                            else:
                                ALL_res.pop(b)
                                xyxy = m_xyxy[b]
                                m_xyxy.pop(b)
                                AISData.pop(AIS_index)
                                for ss in ALL_res:
                                    ss.pop(AIS_index)

                            # im0 = plot_one_box(xyxy, im0, label=AIS_mmsi, color=(0, 0, 255), line_thickness=3)

                            if AIS_mmsi == "NO MATCH":

                                # 获取游标对象
                                insert_sql = """INSERT INTO camera_result(reset_time,minU,minV,maxU,maxV,match_label,ship_number)
                                                VALUES ('%s',%d,%d,%d,%d,0,%d)
                                                """ % (now_time, xyxy[0], xyxy[1], xyxy[2], xyxy[3], ship_number)
                                LOGGER.info('未匹配上')

                            else:
                                a11111 = 1
                                # 获取游标对象
                                insert_sql = """
                                          INSERT INTO camera_result(reset_time,
                                          mmsi,lat,lon,name,customName,heading,course,speed,posTime,status,customStatus,shipType,shipTypeSpec,customShipType,breadth,length,sourceId,safeRadius,
                                          ais_minU,ais_minV,ais_maxU,ais_maxV,
                                          minU,minV,maxU,maxV,
                                          p,t,z,
                                          match_label,ship_number)
                                              VALUES ('%s',
                                              %d,%d,%d,'%s','%s',%d,%d,%d,%d,%d,'%s',%d,'%s','%s','%s','%s','%s','%s',
                                              %d,%d,%d,%d,
                                              %d,%d,%d,%d,
                                              %f,%f,%f,
                                              %d,%d)
                                          """ % (now_time,
                                                 sqlAIS["mmsi"], sqlAIS["lat"], sqlAIS["lon"], sqlAIS["name"],
                                                 sqlAIS["customName"], sqlAIS["heading"], sqlAIS["course"],
                                                 sqlAIS["speed"], sqlAIS["posTime"], sqlAIS["status"],
                                                 sqlAIS["customStatus"], sqlAIS["shipType"], sqlAIS["shipTypeSpec"],
                                                 sqlAIS["customShipType"], sqlAIS["breadth"], sqlAIS["length"],
                                                 sqlAIS["sourceId"], sqlAIS["safeRadius"],
                                                 sqlAIS["minU"], sqlAIS["minV"], sqlAIS["maxU"], sqlAIS["maxV"],
                                                 xyxy[0], xyxy[1], xyxy[2],xyxy[3],
                                                 PTZ["p"],PTZ["t"],PTZ["z"],
                                                 a11111,ship_number)
                                LOGGER.info('数据匹配上了')
                            Insert_Sql(now_time,host, username, password, db_name, insert_sql)
                    else:
                        for xyxy in ALL_xyxy:
                            # im0 = plot_one_box(xyxy, im0, label="NO MATCH", color=(0, 0, 255), line_thickness=3)
                            insert_sql = """INSERT INTO camera_result(reset_time,minU,minV,maxU,maxV,match_label,ship_number)
                                                                            VALUES ('%s',%d,%d,%d,%d,0,%d)
                                                                            """ % (now_time, xyxy[0], xyxy[1], xyxy[2], xyxy[3], ship_number)
                            LOGGER.info('不存在AIS数据')
                            Insert_Sql(now_time,host, username, password, db_name, insert_sql)


def main():
    # 检查依赖包
    check_requirements(exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    run(
        source='https://open.ys7.com/v3/openlive/G18183870_1_2.m3u8?expire=1677567819&id=420234950509830144&t=c784a19375d066e715d175a1d68ad8146b88a84447e03c7bb751f02c43aceb0b&ev=100',
        weights='boat428.pt')
