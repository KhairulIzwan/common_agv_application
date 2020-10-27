#!/usr/bin/env python

################################################################################
## {Description}: Person Tracking Deep Learning
################################################################################
## Author: Khairul Izwan Bin Kamsani
## Version: {1}.{0}.{0}
## Email: {wansnap@gmail.com}
################################################################################

"""
Image published (CompressedImage) from tello originally size of 960x720 pixels
We will try to resize it using imutils.resize (with aspect ratio) to width = 320
and then republish it as Image
"""

# import the necessary Python packages
from __future__ import print_function
import sys
import cv2
import time
import numpy as np
import imutils
import random
import apriltag

# import the necessary ROS packages
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Int64
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

import rospy

from common_agv_application.msg import objCenter as objCoord
from common_agv_application.msg import personID
from common_agv_application.msg import centerID
#from common_agv_application.msg import depthID

class PersonTracking:
	def __init__(self):
		# Initialize
		self.bridge = CvBridge()
		self.objectCoord = objCoord()
		self.depthCoord = Int64()
		self.trackingMode = Bool()

		self.boolID_received = False
		self.image_depth_received = False

		rospy.logwarn("Person Tracking Node [ONLINE]...")

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

		# Subscribe to Bool msg
		self.boolID_topic = "/person/bool"
		self.boolID_sub = rospy.Subscriber(
					self.boolID_topic, 
					Bool, 
					self.cbBoolID
					)

		# Subscribe to personID msg
		self.personID_topic = "/person/ID"
		self.personID_sub = rospy.Subscriber(
					self.personID_topic, 
					personID, 
					self.cbPersonID
					)

		# Subscribe to centerID msg
		self.centerID_topic = "/person/center"
		self.centerID_sub = rospy.Subscriber(
					self.centerID_topic, 
					centerID, 
					self.cbCenterID
					)

		# Subscribe to Image msg
		self.image_depth_topic = "/camera/depth/image_raw"
		self.image_depth_sub = rospy.Subscriber(
						self.image_depth_topic, 
						Image, self.cbImageDepth
						)

		# Subscribe to CameraInfo msg
		self.cameraInfo_depth_topic = "/camera/depth/camera_info"
		self.cameraInfo_depth_sub = rospy.Subscriber(
						self.cameraInfo_depth_topic, 
						CameraInfo, 
						self.cbCameraInfoDepth
						)

		# Publish to objCenter msg
		self.objCoord_topic = "/person/objCoord"
		self.objCoord_pub = rospy.Publisher(
					self.objCoord_topic, 
					objCoord, 
					queue_size=10
					)

		# Publish to Int64 msg
		self.depthCoord_topic = "/person/depth"
		self.depthCoord_pub = rospy.Publisher(
					self.depthCoord_topic, 
					Int64, 
					queue_size=10
					)

		# Publish to Bool msg
		self.trackingMode_topic = "/person/tracking"
		self.trackingMode_pub = rospy.Publisher(
					self.trackingMode_topic, 
					Bool, 
					queue_size=10
					)

		# Allow up to one second to connection
		rospy.sleep(1)

	# 
	def cbBoolID(self, msg):
		try:
			self.boolID = msg.data
		except KeyboardInterrupt as e:
			print(e)

		if self.boolID is not None:
			self.boolID_received = True
		else:
			self.boolID_received = False

	# 
	def cbPersonID(self, msg):
		self.personID = msg.N

	# 
	def cbCenterID(self, msg):
		self.centerID_X = msg.centerX
		self.centerID_Y = msg.centerY

	# Convert image to OpenCV format
	def cbImageDepth(self, msg):
		try:
#			self.cv_image_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
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

	# Image information callback
	def cbInfo(self):

		fontFace = cv2.FONT_HERSHEY_PLAIN
		fontScale = 0.7
		color = (255, 255, 255)
		colorPose = (0, 0, 255)
		colorIMU = (255, 0, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

	# Show the output frame
	def cbShowImage(self):
		self.image_resized = imutils.resize(self.cv_image_depth, width=300)

		cv2.imshow("Person Detection [Depth]", self.image_resized)
		cv2.waitKey(1)

	#
	def cbPersonTracking(self):
		if self.boolID_received and self.image_depth_received:
			self.cbShowImage()
			if self.boolID:
				# TODO: Which ID to select?
				if not self.personID:
					self.trackingMode.data = False
					pass
				else:
					if self.personID[0] == 0:
						self.objectCoord.centerX = int(self.centerID_X[0])
						self.objectCoord.centerY = int(self.centerID_Y[0])

						# TODO:
						self.depthCoord.data = self.cv_image_depth[self.centerID_X[0], self.centerID_Y[0]]
						self.trackingMode.data = True
					else:
						self.trackingMode.data = False
						pass
			else:
				self.objectCoord.centerX = self.imgWidth_depth // 2
				self.objectCoord.centerY = self.imgHeight_depth // 2

				self.depthCoord.data = self.cv_image_depth[self.imgWidth_depth // 2, self.imgHeight_depth // 2] 
				self.trackingMode.data = False

			self.objCoord_pub.publish(self.objectCoord)
			self.depthCoord_pub.publish(self.depthCoord)
			self.trackingMode_pub.publish(self.trackingMode)

		else:
			rospy.logerr("Please run required node!")

	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("Person Tracking Node [OFFLINE]...")

if __name__ == '__main__':

	# Initialize
	rospy.init_node('person_tracking', anonymous=False)
	person = PersonTracking()
	
	r = rospy.Rate(10)
	
	# Camera preview
	while not rospy.is_shutdown():
		person.cbPersonTracking()
		r.sleep()
