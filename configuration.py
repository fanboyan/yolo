# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 9:35
# @Author  : YAN
# @FileName: configuration.py
# @Software: PyCharm
# @Email   : 812008450@qq.com
# 配置文件
import os


class Config(object):  # 默认配置
    DEBUG = False

    # get attribute
    def __getitem__(self, key):
        return self.__getattribute__(key)


class ProductionConfig(Config):  # 生产环境
    host = "localhost"
    username = "root"
    password = "Aa123456**"
    db_name = "camera_prod"
    port = "8089"


class DevelopmentConfig(Config):  # 开发环境
    host = "localhost"
    username = "root"
    password = "123456"
    db_name = "camera"
    port = "8088"


class TestConfig(Config):  # 测试环境
    host = "localhost"
    username = "root"
    password = "Aa123456**"
    db_name = "camera_test"
    port = "8088"


# 环境映射关系
mapping = {
    'dev': DevelopmentConfig,
    'prod': ProductionConfig,
    'test': TestConfig,
}

# # 一键切换环境
APP_ENV = os.environ.get('APP_ENV', 'dev').lower()  # 设置环境变量为default
config = mapping[APP_ENV]()  # 获取指定的环境
