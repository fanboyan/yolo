import pymysql
from configuration import config
from utils.Match import *
from utils.torch_utils import select_device, time_sync
import os
host = config.host
username = config.username
password = config.password
db_name = config.db_name
start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
camera_id=1
# update_sql="update equipment set visibility=0 where id= %d"%(camera_id)
# Update_Sql(start_time, host, username, password, db_name, update_sql)
un_time=1646
xyxy=[1,2,3,4]
# select_sql = "SELECT ship_height FROM equipment WHERE id = %s" % (camera_id)
# ship_height = SELECT_Sql(host, username, password, db_name, select_sql)
# print(ship_height[0][0])
# baseline_id=29
# for i in range(30 - baseline_id):
#     print(i)
# now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H:%M:%S')
# now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H-%M-%S')
# saveimages=f'./images/{now_time}.jpg'
# print(saveimages)
# import cv2

# cv2.imwrite(saveimages, img3)
# 创建文件夹
# period = datetime.datetime.now()
# timestr = period.strftime("%Y_%m_%d")
# savedir_path = './images/' + str(timestr)
# if not os.path.exists(savedir_path):
#     os.makedirs(savedir_path)
# imgFile = "./images/1.jpg"  # 读取文件的路径
# img3 = cv2.imread(imgFile)  # flags=1 读取彩色图像(BGR)
# now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H-%M-%S')
# saveimages = f'{savedir_path}/{now_time}.jpg'
# cv2.imwrite(saveimages, img3)
# print(saveimages)
# t1=time_sync()
# print(time_sync()%1)
# aaa=str(round(time_sync()%1,2)*100)
# now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H-%M-%S')
# print(f'/{now_time}-{aaa}.jpg')


import cv2
import numpy as np

img = np.zeros([600, 600, 3])
points = np.array([[-100, 200], [200, 300], [330, 100], [340, 300], [340, 200], [270, 130]], np.int32)
img = cv2.polylines(img, [points], isClosed=True, color=[0, 0, 255], thickness=5)
img = cv2.fillPoly(img, [points], color=[0, 255, 0])
cv2.imwrite("a.jpg", img)
