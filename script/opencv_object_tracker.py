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
from imutils.video import FPS
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

class OpenCVObjectTracker:
	def __init__(self):
		# initialize our bridge
		self.bridge = CvBridge()

		self.image_received = False

		rospy.logwarn("OpenCVObjectTracker Node [ONLINE]...")

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

		# extract the OpenCV version info
		(major, minor) = cv2.__version__.split(".")[:2]

		self.tracker_method = "kcf"
		
		# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
		# function to create our object tracker
		if int(major) == 3 and int(minor) < 3:
			self.tracker = cv2.Tracker_create(self.tracker_method.upper())
		# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
		# approrpiate object tracker constructor:
		else:
			# initialize a dictionary that maps strings to their corresponding
			# OpenCV object tracker implementations
			OPENCV_OBJECT_TRACKERS = {
#				"csrt": cv2.TrackerCSRT_create,
				"kcf": cv2.TrackerKCF_create,
				"boosting": cv2.TrackerBoosting_create,
				"mil": cv2.TrackerMIL_create,
				"tld": cv2.TrackerTLD_create,
				"medianflow": cv2.TrackerMedianFlow_create,
				"mosse": cv2.TrackerMOSSE_create
			}
			# grab the appropriate object tracker using our dictionary of
			# OpenCV object tracker objects
			self.tracker = OPENCV_OBJECT_TRACKERS[self.tracker_method]()

		# initialize the bounding box coordinates of the object we are going
		# to track
		self.initBB = None

		# initialize the FPS throughput estimator
		self.fps = None

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

	# Object Tracker callback
	def cbOpenCVObjectTracker(self):
		# resize the frame (so we can process it faster) and grab the
		# frame dimensions
		self.image = imutils.resize(self.cv_image.copy(), width=300)
		(H, W) = self.image.shape[:2]

		# check to see if we are currently tracking an object
		if self.initBB is not None:
			# grab the new bounding box coordinates of the object
			(self.success, self.box) = self.tracker.update(self.image)

			# check to see if the tracking was a success
			if self.success:
				(self.x, self.y, self.w, self.h) = [int(v) for v in self.box]
				cv2.rectangle(self.image, (self.x, self.y), (self.x + self.w, self.y + self.h),
					(0, 255, 0), 2)

			# update the FPS counter
			self.fps.update()
			self.fps.stop()

			# initialize the set of information we'll be displaying on
			# the frame
			info = [
				("Tracker", self.tracker_method),
				("Success", "Yes" if self.success else "No"),
				("FPS", "{:.2f}".format(self.fps.fps())),
			]

			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(self.image, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# Show the output frame
	def cbShowImage(self):
		cv2.imshow("Tracker", self.image)
		cv2.waitKey(1)
		key = cv2.waitKey(1) & 0xFF

		# if the 's' key is selected, we are going to "select" a bounding
		# box to track
		if key == ord("s"):
			# select the bounding box of the object we want to track (make
			# sure you press ENTER or SPACE after selecting the ROI)
			self.initBB = cv2.selectROI("Tracker", self.image, fromCenter=False,
				showCrosshair=True)

			# start OpenCV object tracker using the supplied bounding box
			# coordinates, then start the FPS throughput estimator as well
			self.tracker.init(self.image, self.initBB)
			self.fps = FPS().start()

		# if the `q` key was pressed, break from the loop
		elif key == ord("q"):
			self.initBB = None

	# Preview image + info
	def cbPreview(self):
		if self.image_received:
			self.cbOpenCVObjectTracker()
			self.cbShowImage()
		else:
			rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("OpenCVObjectTracker Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initialize
	rospy.init_node('object_tracker', anonymous=False)
	obj = OpenCVObjectTracker()
	
#	r = rospy.Rate(10)

	# OpenCVObjectTracker
	while not rospy.is_shutdown():
		obj.cbPreview()
#		r.sleep()
