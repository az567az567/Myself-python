### RTSP

import time
import cv2 as cv
import os
import numpy as np
import argparse

Conf_threshold = 0.4 #置信度阀值
NMS_threshold = 0.4 #非极大值抑制阀值
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)] #颜色

class_names = [] #初始化一个列表以存储类名
with open("C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\fire.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#opencv的dnn模块(NVIDIA GPU的推理模块)
net = cv.dnn.readNet("C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\weights\\yolov4-fire_last.weights",
					 "C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\yolov4-fire.cfg")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

"""
cam = cv.VideoCapture('rtsp://adminn:098765@192.168.43.50:554/stream1')

while True:
	ret, img = cam.read()
	vis = img.copy()
	cv.imshow('getCamera',vis)
	if 0xFF & cv.waitKey(5) == 27:
		break

cv.destroyAllWindows()
"""

RTSP_URL = 'rtsp://adminn:098765@192.168.43.50:554/stream1'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv.VideoCapture(RTSP_URL)

if not cap.isOpened():
	print('Cannot open RTSP stream')
	exit(-1)

while True:
	_, frame = cap.read()
	cv.imshow('RTSP stream', frame)

	if cv.waitKey(1) == 27:
		break

cap.release()
cv.destroyAllWindows()