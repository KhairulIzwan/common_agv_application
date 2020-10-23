#!/usr/bin/env python

################################################################################
## {Description}: 
################################################################################
## Author: Khairul Izwan Bin Kamsani
## Version: {1}.{0}.{0}
## Email: {wansnap@gmail.com}
################################################################################

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
from geometry_msgs.msg import Twist

import rospy

from common_agv_application.pid import PID
from common_agv_application.makesimpleprofile import map as mapped

from common_agv_application.msg import objCenter as objCoord
from common_agv_application.msg import personID
from common_agv_application.msg import centerID
#from common_agv_application.msg import depthID

class PersonFollow:
	def __init__(self):
		# Initialize
		self.robotCmdVel = Twist()

		self.trackingMode_received = False

		self.MAX_LIN_VEL = 2.00
		self.MAX_ANG_VEL = 0.4

		# set PID values for pan
		self.panP = 0.5
		self.panI = 0
		self.panD = 0

		# set PID values for tilt
		self.tiltP = 1
		self.tiltI = 0
		self.tiltD = 0

		# create a PID and initialize it
		self.panPID = PID(self.panP, self.panI, self.panD)
		self.tiltPID = PID(self.tiltP, self.tiltI, self.tiltD)

		self.panPID.initialize()
		self.tiltPID.initialize()

		rospy.logwarn("AprilTag Tracking Node [ONLINE]...")

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

		# Subscribe to CameraInfo msg
		self.cameraInfo_depth_topic = "/camera/depth/camera_info"
		self.cameraInfo_depth_sub = rospy.Subscriber(
						self.cameraInfo_depth_topic, 
						CameraInfo, 
						self.cbCameraInfoDepth
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

		# Subscribe to Bool msg
		self.trackingMode_topic = "/person/tracking"
		self.trackingMode_sub = rospy.Subscriber(
					self.trackingMode_topic, 
					Bool, 
					self.cbTrackingMode
					)

		# Publish to Twist msg
		self.robotCmdVel_topic = "/cmd_vel"
		self.robotCmdVel_pub = rospy.Publisher(
					self.robotCmdVel_topic, 
					Twist, 
					queue_size=10
					)

		# Allow up to one second to connection
		rospy.sleep(1)

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

	# show information callback
	def cbPIDerr(self):
		self.panErr, self.panOut = self.cbPIDprocess(self.panPID, self.objCoord_X, self.imgWidth_depth // 2)
#		self.tiltErr, self.tiltOut = self.cbPIDprocess(self.tiltPID, self.objCoord_Y, self.imgHeight_depth // 2)
		self.tiltErr, self.tiltOut = self.cbPIDprocess(self.tiltPID, self.objCoord_depth, 1500)

	def cbPIDprocess(self, pid, objCoord, centerCoord):
		# calculate the error
		error = centerCoord - objCoord

		# update the value
		output = pid.update(error)

		return error, output

	def cbCallErr(self):
		# Is self.trackingMode_received: True?
		if self.trackingMode_received:
			# Is self.trackingMode: True?
			if self.trackingMode:
				# Start trackingMode
				self.cbPIDerr()

				panSpeed = mapped(abs(self.panOut), 0, self.imgWidth_depth // 2, 0, self.MAX_ANG_VEL)
#				tiltSpeed = mapped(abs(self.tiltOut), 0, self.imgHeight // 2, 0, self.MAX_LIN_VEL)
				tiltSpeed = mapped(abs(self.tiltOut), 0, 1500, 0, self.MAX_LIN_VEL)

				if self.panOut < 0:
					self.robotCmdVel.angular.z = panSpeed
				elif self.panOut > 0:
					self.robotCmdVel.angular.z = -panSpeed
				else:
					self.robotCmdVel.angular.z = 0

				if self.tiltOut > 0:
					self.robotCmdVel.linear.x = tiltSpeed
				elif self.tiltOut < 0:
					self.robotCmdVel.linear.x = -tiltSpeed
				else:
					self.robotCmdVel.linear.x = 0

#				self.robotCmdVel.linear.x = 0
				self.robotCmdVel.linear.y = 0
				self.robotCmdVel.linear.z = 0

				self.robotCmdVel.angular.x = 0.0
				self.robotCmdVel.angular.y = 0.0
#				self.robotCmdVel.angular.z = 0.0

				self.robotCmdVel_pub.publish(self.robotCmdVel)
				
			# Is self.trackingMode: False?
			else:
				# trackingMode Halt!
				self.robotCmdVel.linear.x = 0.0
				self.robotCmdVel.linear.y = 0.0
				self.robotCmdVel.linear.z = 0.0

				self.robotCmdVel.angular.x = 0.0
				self.robotCmdVel.angular.y = 0.0
				self.robotCmdVel.angular.z = 0.0
				self.robotCmdVel_pub.publish(self.robotCmdVel)

		# Is self.trackingMode_received: False
		else:
			pass

	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("AprilTag Tracking Node [OFFLINE]...")

if __name__ == '__main__':

	# Initialize
	rospy.init_node('camera_apriltag_tracking', anonymous=False)
	camera = PersonFollow()
	
	r = rospy.Rate(10)
	
	# Camera preview
	while not rospy.is_shutdown():
		camera.cbCallErr()
		r.sleep()
