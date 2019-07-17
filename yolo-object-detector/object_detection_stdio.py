#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from ctypes import *
import math
import random
import cv2
import time , threading
import json, sys, os, signal
import numpy as np
import darknet
from sort import *

def shutdown(self, signum):
	cap.release()
	to_node("status", 'Shutdown: Done.')
	exit()

def to_node(type, message):
	# convert to json and print (node helper will read from stdout)
	try:
		print(json.dumps({type: message}))
	except Exception:
		pass
	# stdout has to be flushed manually to prevent delays in the node helper communication
	sys.stdout.flush()

FPS = 5.

def check_stdin():
	global FPS

	while True:
		lines = sys.stdin.readline()
		data = json.loads(lines)
		if 'FPS' in data:
			FPS = data['FPS']

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def convertToCenterHW(a,b,c,d):
	h = float(d - b)
	w = float(c - a)
	x = float((a + (w/2)) / 1080)
	y = float((b + (h/2)) / 1920)

	return (x,y),(w/1080,h/1920)


if __name__ == "__main__":


	BASE_DIR = os.path.dirname(__file__) + '/'
	os.chdir(BASE_DIR)


	to_node("status", "Object detection is starting...")

	""" 
	get image from gestreamer appsink!
	"""
	cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR, width=1080, height=1920, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", cv2.CAP_GSTREAMER)
	#cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_1m ! video/x-raw, format=BGR, width=1080, height=1920, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", cv2.CAP_GSTREAMER)
	#cap = cv2.VideoCapture(3)
	#cap.set(3,1920);
	#cap.set(4,1080);
	#cv2.namedWindow("objects recognition tracked", cv2.WINDOW_NORMAL)

	"""
	preparare darknet neural network for hand object recognition
	"""

	darknet.set_gpu(1)

	configPath = "cfg/yolov3.cfg"
	weightPath = "data/yolov3.weights"
	metaPath = "data/coco.data"


	thresh = 0.7
	hier_thresh=.45
	nms=.45 
	debug= False

	netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
	metaMain = darknet.load_meta(metaPath.encode("ascii"))

	"""
	start thread for standart in
	"""	
	t = threading.Thread(target=check_stdin)
	t.start()
	

	"""
	in case of shutdown
	"""
	signal.signal(signal.SIGINT, shutdown)

	#raster for hand tracing.. here the image resolution 
	horizontal_division = 180.0
	vertical_division =  320.0

	DetectionArray = np.zeros((int(vertical_division),int(horizontal_division),metaMain.classes),dtype=np.uint8)

	to_node("status", "Object detection started...")

	darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)

	last_detection_list = []

	while True:

		start_time = time.time()

		ret, frame = cap.read()
		if ret is False:
			continue

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)


		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

		dets = darknet.detect_image(netMain, metaMain, darknet_image, thresh=thresh)

		detection_list = []

		for det in dets:
			x, y, w, h = det[2][0],\
            det[2][1],\
            det[2][2],\
            det[2][3]

			xrel = float(int(float(x / darknet.network_width(netMain)) * horizontal_division) / horizontal_division)
			yrel = float(int(float(y / darknet.network_height(netMain)) * vertical_division) / vertical_division)
			w = float(w / darknet.network_width(netMain))
			h = float(h / darknet.network_height(netMain))

			for j in range(metaMain.classes):
				if det[0] == metaMain.names[j]:
					i = j
					break

			detection_list.append({ "name": metaMain.names[i].decode('utf-8'), "center": (float("{0:.5f}".format(xrel)), float("{0:.5f}".format(yrel))), "w_h": (float("{0:.5f}".format(w)),float("{0:.5f}".format(h)))  })

		if not(not last_detection_list and not detection_list):		
	
			equality_counter = 0
			for prev_element in last_detection_list:
				for next_element in detection_list:
					if next_element["center"] == prev_element["center"] and next_element["name"] == prev_element["name"]:
						equality_counter += 1
			
			if not (equality_counter == len(last_detection_list) == len(detection_list)):
				to_node("DETECTED_OBJECTS",detection_list)
				last_detection_list = detection_list
					
				

		delta = time.time() - start_time
		if (1.0 / FPS) - delta > 0:
			time.sleep((1.0 / FPS) - delta)
			fps_cap = FPS
		else:
			fps_cap = 1. / delta

		to_node("status", fps_cap)

		#cv2.putText(frame, str(round(fps_cap)) + " FPS", (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(50,255,50), thickness=3)

		#cv2.imshow("objects recognition tracked", frame)

		#out_cap.write(image_cap)
		#out_indicator.write(image_indicator)
	
		#cv2.waitKey(33)
