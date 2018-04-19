import numpy as npy
import cv2
import os
import re
import time

'''
star和sift结合的特征检测
返回值：
points特征点对象列表
features特征向量
'''

def get_features(image_path):
    current_img = cv2.imread(image_path)
    # 定义为相同大小
    current_img = cv2.resiz(current_img, (800, 800))
    # 先转化为灰度图,然后直方图均衡化,目的是增加对比度
    gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    gray_img_hist = cv2.equalizeHist(gray_img)
    start = cv2.xfeatures2d.StarDetector_create()
    # 先通过Stat模型进行一次特征点检测
    point = start.detect(gray_img_hist)
    sift = cv2.xfeatures2d.SIFT_create()
    # 通过compute对之前的检测器所获得的特征点进行迭代
    points, features = sift.compute(gray_img_hist, point)
    return points, features


path = "./source"
dir_list = os.listdir(path)
for d in dir_list:
    if os.path.isdir(path + "/" + d):
        file_list = os.listdir(path + "/" + d)
        for f in file_list:
		    # 只获取当前文件夹下的图片文件
            if re.search("\\w+\\.(jpg|gif|bmp|png)$",f) is None:
                continue
            else:
                image_path = path + "/" + d + "/" + f
                t1 = time.time()
                kps, fes = get_features(image_path)
                t2 = time.time()
                print("类别为"+d+"的图片"+f+"提取特征点数为："+str(len(kps))+"  耗时："+str(round((t2-t1)*1000))+"毫秒")