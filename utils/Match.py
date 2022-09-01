import datetime
import requests
import json
from numpy import *
from random import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymysql
import time
from utils.general import LOGGER

# 相机预测距离
def distance_to_camera(knownWidth=30, focalLength=449.007, perWidth=50, Z=1):
    """
    knownWidth：知道的目标宽度 厘米
    focalLength：摄像头焦距
    perWidth：检测框宽度  像素

    """
    return (knownWidth * focalLength * Z) / perWidth

def height_to_camera(perHeight,fy,distance,Z):
    perHeight=perHeight
    if Z>=30:
        Z=30
    f=fy*Z

    return perHeight*distance/f
def height_to_camera1(perHeight,perWidth,ship_Width):


    return perHeight*ship_Width/perWidth
# 计算两经纬度点之间距离
def getDistance(latA, lonA, latB, lonB):
    # 计算两点距离
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance = distance
    return distance  # m


def GetAngle(line1, line2):
    """
    计算两条线段之间的夹角
    :param line1:
    :param line2:
    :return:
    """
    dx1 = line1.Point1.X - line1.Point2.X
    dy1 = line1.Point1.Y - line1.Point2.Y
    dx2 = line2.Point1.X - line2.Point2.X
    dy2 = line2.Point1.Y - line2.Point2.Y
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    return insideAngle


# 画框
def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        font_size = t_size[1]
        font = ImageFont.truetype('simhei.ttf', font_size)
        t_size = font.getsize(label)
        c2 = c1[0] + t_size[0], c1[1] - t_size[1]
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        draw.text((c1[0], c2[1] - 2), label, fill=(255, 255, 255), font=font)

        return cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)


# 得到当前PTZ 卡达凯斯
def GetPtz():
    url = "http://kdks.cniship.com:7080/ship-client/index/getPTZ"
    try:
        # get请求方式
        respones = requests.get(url=url)
    except Exception as e:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        LOGGER.info(f'{str(now_time)} : 拿取PTZ失败\n{e}')
    # get请求方式
    data = respones.text
    # 将 JSON 对象转换为 Python 字典

    data = json.loads(data)

    PTZ = data['result']
    return PTZ


# setPTZ 卡达凯斯
def SetPtz(P, T, Z):
    url = "http://kdks.cniship.com:7080/ship-client/index/setPTZ?range=" + P + "&tilt=" + T + "&multiple=" + Z

    try:
        # get请求方式
        respones = requests.get(url=url)
    except Exception as e:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        LOGGER.info(f'{str(now_time)} : 设置PTZ失败\n{e}')
    data = respones.text
    # 将 JSON 对象转换为 Python 字典
    data = json.loads(data)

    ALL_AIS = data['message']
    return ALL_AIS

# 在图片中画区域
def pltarea(img, pts, color, thickness):
    # pts为点的的list
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

# 拿到AIS数据
def getAISData(port):
    url = "http://localhost:" + port + "/latestuv"

    try:
        # get请求方式
        response = requests.get(url=url)
    except Exception as url_error:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        ALL_AIS = []
        LOGGER.info(f'{str(now_time)} : 无法连接java后端，拿取AIS\n{url_error}')
    else:
        data = response.text
        # 将 JSON 对象转换为 Python 字典
        data = json.loads(data)
        try:
            ALL_AIS = data['result']
            pass
        except Exception as data_error:
            now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
            ALL_AIS = []
            LOGGER.info(f'{str(now_time)} : 获取AIS失败\n{data_error}')
    return ALL_AIS


# 拿到PTZ
def GainPtz(port):
    url = "http://localhost:" + port + "/gainPtz"
    try:
        # get请求方式
        response = requests.get(url=url)
    except Exception as url_error:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        PTZ = {'p': 0, 't': 0, 'z': 0}
        LOGGER.info(f'{str(now_time)} : 无法连接java后端，拿取PTZ\n{url_error}')
    else:
        # get请求方式
        data = response.text
        # 将 JSON 对象转换为 Python 字典
        data = json.loads(data)
        try:
            PTZ = data['result']
            pass
        except Exception as data_error:
            now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
            PTZ = {'p': 0, 't': 0, 'z': 0}
            LOGGER.info(f'{str(now_time)} : 获取PTZ失败\n{data_error}')
    return PTZ


# 得到当前PTZ 卡达凯斯
def GetPtz(camera_id=None):
    if camera_id==2:
        url = "http://kdks.cniship.com:7080/ship-client/index/getPTZ"
    elif camera_id==1:
        url = "http://yuandang.cniship.com:7889/ship-client/index/getCameraPTZ?channel=1"
    try:
        # get请求方式
        respones = requests.get(url=url)
    except Exception as e:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        LOGGER.info(f'{str(now_time)} : 拿取PTZ失败\n{e}')
    # get请求方式
    data = respones.text
    # 将 JSON 对象转换为 Python 字典

    data = json.loads(data)

    PTZ = data['result']
    return PTZ


# setPTZ 卡达凯斯
def SetPtz(P, T, Z):
    url = "http://kdks.cniship.com:7080/ship-client/index/setPTZ?range=" + P + "&tilt=" + T + "&multiple=" + Z

    try:
        # get请求方式
        respones = requests.get(url=url)
    except Exception as e:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        LOGGER.info(f'{str(now_time)} : 设置PTZ失败\n{e}')
    data = respones.text
    # 将 JSON 对象转换为 Python 字典
    data = json.loads(data)

    ALL_AIS = data['message']
    return ALL_AIS


# 拿到相机时间
def GainTime(port):
    url = "http://localhost:" + port + "/gainTime"
    try:
        # get请求方式
        response = requests.get(url=url)
    except Exception as url_error:
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        camera_time = 0
        LOGGER.info(f'{str(now_time)} : 无法连接java后端，拿取相机时间\n{url_error}')
    else:
        data = response.text
        # 将 JSON 对象转换为 Python 字典
        data = json.loads(data)
        try:
            camera_time = data['result']
        except Exception as data_error:
            now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
            camera_time = 0
            LOGGER.info(f'{str(now_time)} : 拿取相机时间失败\n{data_error}')
    # 将时间戳转为时间格式
    result = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(camera_time))  # str格式
    # result = datetime.datetime.strptime(result, '%Y-%m-%d %H:%M:%S')  # 时间格式
    # 1970-01-01 08:00:00 错误数据
    return result



# 插入数据库
def Insert_Sql(now_time, host, username, password, db_name, insert_sql):
    connect = pymysql.connect(host=host, user=username, password=password, db=db_name, charset='utf8')

    # 获取游标对象
    cursor = connect.cursor()

    try:
        # 插入数据
        cursor.execute(insert_sql)
        connect.commit()
        LOGGER.info(f'{now_time},插入成功')
    except Exception as e:
        connect.rollback()
        LOGGER.info(f'{now_time},插入失败, {e}')
    cursor.close()
    connect.close()


# 修改数据库数据
def Update_Sql(now_time, host, username, password, db_name, update_sql):
    connect = pymysql.connect(host=host, user=username, password=password, db=db_name, charset='utf8')

    # 获取游标对象
    cursor = connect.cursor()

    # # 将时间戳转为时间格式
    # dateTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    try:
        # 修改数据
        cursor.execute(update_sql)
        connect.commit()
        LOGGER.info(f'{now_time},修改成功')
    except Exception as e:
        connect.rollback()
        LOGGER.info(f'{now_time},修改失败, {e}')
    cursor.close()
    connect.close()


# 查询数据库
def SELECT_Sql(host, username, password, db_name, select_sql):
    connect = pymysql.connect(host=host, user=username, password=password, db=db_name, charset='utf8')

    # 获取游标对象
    cursor = connect.cursor()
    # SQL 查询语句
    try:
        # 执行SQL语句
        cursor.execute(select_sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        results = list(results)
    except Exception as e:
        connect.rollback()
        print("查询失败",e)
    cursor.close()
    connect.close()
    return results

#循环插入剩余
def Remain_buy_insert(camera_id,start_time,baseline_id,visibility_id,host,username,password,db_name):
    for i in range(30 - baseline_id):
        insert_sql = """INSERT INTO visibility_result(camera_id,set_time,baseline_id,visibility_id,is_visibility)
                                                                                                        VALUES (%d,'%s',%d,%d,0)
                                                                                                        """ % (
            camera_id, start_time, baseline_id+i+1, visibility_id)
        Insert_Sql(start_time, host, username, password, db_name, insert_sql)
    return 1

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=config.port, debug=True)