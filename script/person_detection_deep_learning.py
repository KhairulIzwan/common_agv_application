#!/usr/bin/env python

################################################################################
## {Description}: Person Detection Deep Learning (MobileNetSSD)
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
from std_msgs.msg import Bool
from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

import rospy
import rospkg

from common_agv_application.centroidtracker import CentroidTracker

from common_agv_application.msg import objCenter as objCoord
from common_agv_application.msg import personID
from common_agv_application.msg import centerID
#from common_agv_application.msg import depthID

class PersonDetection:
	def __init__(self):
		# initialize our centroid tracker, bridge, and rospack
		self.ct = CentroidTracker()

		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()

		self.boolID = Bool()
		self.personID = personID()
		self.centerID = centerID()
#		self.depthID = depthID()

		self.image_rgb_received = False
		self.trackingMode_received = False
#		self.image_depth_received = False

		rospy.logwarn("Person Detection Node [ONLINE]...")

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
		self.image_rgb_topic = "/camera/rgb/image_raw"
		self.image_rgb_sub = rospy.Subscriber(
						self.image_rgb_topic, 
						Image, self.cbImageRGB
						)

		# Subscribe to CameraInfo msg
		self.cameraInfo_rgb_topic = "/camera/rgb/camera_info"
		self.cameraInfo_rgb_sub = rospy.Subscriber(
						self.cameraInfo_rgb_topic, 
						CameraInfo, 
						self.cbCameraInfoRGB
						)

#		# Subscribe to Image msg
#		self.image_depth_topic = "/camera/depth/image_raw"
#		self.image_depth_sub = rospy.Subscriber(
#						self.image_depth_topic, 
#						Image, self.cbImageDepth
#						)

#		# Subscribe to CameraInfo msg
#		self.cameraInfo_depth_topic = "/camera/depth/camera_info"
#		self.cameraInfo_depth_sub = rospy.Subscriber(
#						self.cameraInfo_depth_topic, 
#						CameraInfo, 
#						self.cbCameraInfoDepth
#						)

		# TODO:
		# Subscribe to Bool msg
		self.trackingMode_topic = "/person/tracking"
		self.trackingMode_sub = rospy.Subscriber(
					self.trackingMode_topic, 
					Bool, 
					self.cbTrackingMode
					)

		# Subscribe to objCenter msg
		self.objCoord_topic = "/person/objCoord"
		self.objCoord_sub = rospy.Subscriber(
					self.objCoord_topic, 
					objCoord, 
					self.cbObjCoord
					)

		# Subscribe to depthID msg
		self.depthCoord_topic = "/person/depth"
		self.depthCoord_sub = rospy.Subscriber(
					self.depthCoord_topic, 
					Int64, 
					self.cbDepthCoord
					)

		# Publish to Bool msg
		self.boolID_topic = "/person/bool"
		self.boolID_pub = rospy.Publisher(
					self.boolID_topic, 
					Bool, 
					queue_size=10
					)

		# Publish to personID msg
		self.personID_topic = "/person/ID"
		self.personID_pub = rospy.Publisher(
					self.personID_topic, 
					personID, 
					queue_size=10
					)

		# Publish to personID msg
		self.personID_topic = "/person/ID"
		self.personID_pub = rospy.Publisher(
					self.personID_topic, 
					personID, 
					queue_size=10
					)

		# Publish to centerID msg
		self.centerID_topic = "/person/center"
		self.centerID_pub = rospy.Publisher(
					self.centerID_topic, 
					centerID, 
					queue_size=10
					)

#		# Publish to depthID msg
#		self.depthID_topic = "/person/depth"
#		self.depthID_pub = rospy.Publisher(
#					self.depthID_topic, 
#					depthID, 
#					queue_size=10
#					)

		# Publish to Image msg
		self.personImage_topic = "/person/image"
		self.personImage_pub = rospy.Publisher(
					self.personImage_topic, 
					Image, 
					queue_size=10
					)

		# Allow up to one second to connection
		rospy.sleep(1)

	# Convert image to OpenCV format
	def cbImageRGB(self, msg):

		try:
			self.cv_image_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

			# un-comment if the image is mirrored
#			self.cv_image_rgb = cv2.flip(self.cv_image_rgb, 1)
		except CvBridgeError as e:
			print(e)

		if self.cv_image_rgb is not None:
			self.image_rgb_received = True
		else:
			self.image_rgb_received = False

	# Get CameraInfo
	def cbCameraInfoRGB(self, msg):
		self.imgWidth_rgb = msg.width
		self.imgHeight_rgb = msg.height

	# Convert image to OpenCV format
	def cbImageDepth(self, msg):

		try:
			self.cv_image_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

			# un-comment if the image is mirrored
#			self.cv_image_depth = cv2.flip(self.cv_image_depth, 1)
		except CvBridgeError as e:
			print(e)

		if self.cv_image_depth is not None:
			self.image_depth_received = True
		else:
			self.image_depth_received = False

	# Get CameraInfo
	def cbCameraInfoDepth(self, msg):
		self.imgWidth_depth = msg.width
		self.imgHeight_depth = msg.height

	# 
	def cbTrackingMode(self, msg):
		try:
			self.trackingMode = msg.data
		except KeyboardInterrupt as e:
			print(e)

		if self.trackingMode is not None:
			self.trackingMode_received = True
		else:
			self.trackingMode_received = False

	# 
	def cbObjCoord(self, msg):
		self.objCoord_X = msg.centerX
		self.objCoord_Y = msg.centerY

	# 
	def cbDepthCoord(self, msg):
		self.objCoord_depth = msg.data

	# Object Detection callback
	def cbPersonDetection(self):
		# load the input image and construct an input blob for the image
		# by resizing to a fixed 300x300 pixels and then normalizing it
		# (note: normalization is done via the authors of the MobileNet SSD
		# implementation)
		self.image = self.cv_image_rgb.copy()
		(self.h, self.w) = self.image.shape[:2]
		self.blob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
#		rospy.logwarn("Computing Object Detections...")
		self.net.setInput(self.blob)
		self.detections = self.net.forward()
		self.rects = []

		self.personID_array = []

		self.centerID_X_array = []
		self.centerID_Y_array = []

#		self.depthID_array = []

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
				# 
				self.boolID.data = True
				
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
			else:
				# 
				self.boolID.data = False

			self.boolID_pub.publish(self.boolID)
			
			if self.trackingMode_received:
				cv2.putText(
						self.image, 
						"TRACKING MODE: %s" % self.trackingMode, 
						(10, 40),
						cv2.FONT_HERSHEY_SIMPLEX, 
						1, 
						(0, 0, 255), 
						4)
				cv2.putText(
						self.image, 
						"CENTER: (%d, %d)" % (self.objCoord_X, self.objCoord_Y), 
						(10, 80),
						cv2.FONT_HERSHEY_SIMPLEX, 
						1, 
						(0, 0, 255), 
						4)
				cv2.putText(
						self.image, 
						"DEPTH: %d" % self.objCoord_depth, 
						(10, 120),
						cv2.FONT_HERSHEY_SIMPLEX, 
						1, 
						(0, 0, 255), 
						4)
			else:
				pass

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

#			depth = self.cv_image_depth[centroid[0], centroid[1]]

			self.personID_array.append(objectID)

			self.centerID_X_array.append(centroid[0])
			self.centerID_Y_array.append(centroid[1])

#			self.depthID_array.append(depth)

		self.personID.N = self.personID_array
		self.personID_pub.publish(self.personID)

		self.centerID.centerX = self.centerID_X_array
		self.centerID.centerY = self.centerID_Y_array
		self.centerID_pub.publish(self.centerID)

#		self.depthID.depth = self.depthID_array
#		self.depthID_pub.publish(self.depthID)

	# Show the output frame
	def cbShowImage(self):
		self.image_resized = imutils.resize(self.image, width=300)

		cv2.imshow("Person Detection [RGB]", self.image_resized)
		cv2.waitKey(1)

		try:
			self.personImage_pub.publish(self.bridge.cv2_to_imgmsg(self.image_resized, "bgr8"))
		except CvBridgeError as e:
			print(e)

	# Preview image + info
	def cbPreview(self):
		if self.image_rgb_received:
			self.cbPersonDetection()
			self.cbInfo()
			self.cbShowImage()
		else:
			rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("Person Detection Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initialize
	rospy.init_node('person_detection', anonymous=False)
	obj = PersonDetection()
	
	r = rospy.Rate(10)

	# PersonDetection
	while not rospy.is_shutdown():
		obj.cbPreview()
		r.sleep()
