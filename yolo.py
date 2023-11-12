###  lots of ,like webcam,image,vedio

import cv2
import os
import numpy as np 
import argparse
import time
import winsound
import urllib.request #

url='rtsp://adminn:098765@192.168.43.50:554/stream1' #
im=None #

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
parser.add_argument('--im_path', help="Path of url to detect objects", default="none") #
args = parser.parse_args()

#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\weights\\yolov4-fire_last.weights","C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\yolov4-fire.cfg")
	classes = []
	with open("C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\fire.names", "r") as f:
		classes = [line.strip() for line in f.readlines()] 
	
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap

def load_url(im):#
   img_resp=urllib.request.urlopen(url)
   imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
   im = cv2.imdecode(imgnp,-1)
   cv2.imshow('live transmission',im)
   img = cv2.imread(im)
   img = cv2.resize(img, None, fx=0.4, fy=0.4)
   height, width, channels = img.shape
   return img, height, width, channels

#url = 'rtsp://admin:password@192.168.137.132:554/11'
#cap = cv2.VideoCapture(url)


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
			
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
			winsound.Beep(500, 1000)  #
	cv2.imshow("Image", img)

def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	#winsound.Beep(500, 1000)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		#winsound.Beep(500, 1000) #sound
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()

def url_detect(im):#
   model, classes, colors, output_layers = load_yolo()
   image, height, width, channels = load_url(im)
   blob, outputs = detect_objects(image, model, output_layers)
   boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
   draw_labels(boxes, confs, colors, class_ids, classes, image)
   while True:
      key = cv2.waitKey(1)
      if key == 27:
         break

def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		#winsound.Beep(500, 1000)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()



if __name__ == '__main__':
	webcam = args.webcam
	video_play = args.play_video
	image = args.image
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)
	# if url:  #
		im_path = args.im_path
		if args.verbose:
			print('---- Starting urlWebCam object detection ----')
		url_detect(im)

	cv2.destroyAllWindows()


"""


cam = cv2.VideoCapture('rtsp://adminn:098765@192.168.43.50:554/stream1')

while True:
	ret, img = cam.read()
	vis = img.copy()
	cv2.imshow('getCamera',vis)
	if 0xFF & cv2.waitKey(5) == 27:
		break

cv2.destoryAllWindows()


"""

"""

RTSP_URL = 'rtsp://adminn:098765@192.168.43.50:554/stream1'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
	print('Cannot open RTSP stream')
	exit(-1)

while True:
	_, frame = cap.read()
	cv2.imshow('RTSP stream', frame)

	if cv2.waitKey(1) == 27:
		break

cap.release()
cv2.destroyAllWindows()
"""
