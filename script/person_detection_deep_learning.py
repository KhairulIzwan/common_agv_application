#!/usr/bin/env python

################################################################################
## {Description}: Object Detection Deep Learning
################################################################################
## Author: Khairul Izwan Bin Kamsani
## Version: {1}.{0}.{0}
## Email: {wansnap@gmail.com}
################################################################################

# import the necessary Python packages
from __future__ import print_function
from imutils import face_utils
import sys
import cv2
import time
import imutils
import dlib
import os
import numpy as np

# import the necessary ROS packages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

import rospy
import rospkg

from common_agv_application.centroidtracker import CentroidTracker

class PersonDetection:
	def __init__(self):
		# initialize our centroid tracker, bridge, and rospack
		self.ct = CentroidTracker()
		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()

		self.image_received = False

		rospy.logwarn("PersonDetection Node [ONLINE]...")

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

		# initialize the list of class labels MobileNet SSD was trained to
		# detect, then generate a set of bounding box colors for each class
		self.CLASSES = [
			"background", 
			"aeroplane", 
			"bicycle", 
			"bird", 
			"boat",
			"bottle", 
			"bus", 
			"car", 
			"cat", 
			"chair", 
			"cow", 
			"diningtable",
			"dog", 
			"horse", 
			"motorbike", 
			"person", 
			"pottedplant", 
			"sheep",
			"sofa", 
			"train", 
			"tvmonitor"
			]

		self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

		# Load our serialized model from disk
		self.package = os.path.sep.join([self.rospack.get_path('common_agv_application')])
		self.modelDir = os.path.join(self.package, "model")

		# Caffe 'deploy' prototxt file
		self.prototxt = self.modelDir + "/MobileNetSSD_deploy.prototxt.txt"
		# Caffe pre-trained model
		self.model = self.modelDir + "/MobileNetSSD_deploy.caffemodel"

		self.confidenceParam = 0.8

		# load our serialized model from disk
		rospy.loginfo("Loading Model...")
		self.net = cv2.dnn.readNetFromCaffe(
						self.prototxt, 
						self.model
						)

		# Subscribe to Image msg
		self.image_topic = "/camera/rgb/image_raw"
		self.image_sub = rospy.Subscriber(
						self.image_topic, 
						Image, self.cbImage
						)

		# Subscribe to CameraInfo msg
		self.cameraInfo_topic = "/camera/rgb/camera_info"
		self.cameraInfo_sub = rospy.Subscriber(
						self.cameraInfo_topic, 
						CameraInfo, 
						self.cbCameraInfo
						)

		# Allow up to one second to connection
		rospy.sleep(1)

	# Convert image to OpenCV format
	def cbImage(self, msg):

		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		if self.cv_image is not None:
			self.image_received = True
		else:
			self.image_received = False

	# Get CameraInfo
	def cbCameraInfo(self, msg):
		self.imgWidth = msg.width
		self.imgHeight = msg.height

	# Object Detection callback
	def cbPersonDetection(self):
		# load the input image and construct an input blob for the image
		# by resizing to a fixed 300x300 pixels and then normalizing it
		# (note: normalization is done via the authors of the MobileNet SSD
		# implementation)
		self.image = self.cv_image.copy()
		(self.h, self.w) = self.image.shape[:2]
		self.blob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
#		rospy.logwarn("Computing Object Detections...")
		self.net.setInput(self.blob)
		self.detections = self.net.forward()
		self.rects = []

	# Object Detection Information
	def cbInfo(self):
		# loop over the detections
		for i in np.arange(0, self.detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			self.confidence = self.detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if self.confidence > self.confidenceParam:
				# extract the index of the class label from the `detections`,
				# then compute the (x, y)-coordinates of the bounding box for
				# the object
				self.idx = int(self.detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if self.CLASSES[self.idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				self.box = self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
				self.rects.append(self.box.astype("int"))

				(startX, startY, endX, endY) = self.box.astype("int")

				# display the prediction
				self.label = "{}: {:.2f}%".format(self.CLASSES[self.idx], self.confidence * 100)
#					rospy.loginfo("{}".format(self.label))
				cv2.rectangle(
						self.image, 
						(startX, startY), 
						(endX, endY),
						self.COLORS[self.idx], 
						2
						)

				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(
					self.image, 
					self.label, 
					(startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 
					0.5, 
					self.COLORS[self.idx], 
					2)

		# update our centroid tracker using the computed set of bounding
		# box rectangles
		objects = self.ct.update(self.rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(self.image, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(self.image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# Show the output frame
	def cbShowImage(self):
		cv2.imshow("Person Detection", self.image)
		cv2.waitKey(1)

	# Preview image + info
	def cbPreview(self):
		if self.image_received:
			self.cbPersonDetection()
			self.cbInfo()
			self.cbShowImage()
		else:
			rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("PersonDetection Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initialize
	rospy.init_node('person_detection', anonymous=False)
	obj = PersonDetection()
	
#	r = rospy.Rate(10)

	# PersonDetection
	while not rospy.is_shutdown():
		obj.cbPreview()
#		r.sleep()
