#!/usr/bin/env python

################################################################################
## {Description}: Accessing raspicam/usbcam
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
import imutils

# import the necessary ROS packages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

import rospy

class CameraPreview:
	def __init__(self):

		self.bridge = CvBridge()
		self.image_received = False

		rospy.logwarn("CameraPreview [Depth] Node [ONLINE]...")

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

		# Subscribe to Image msg
		self.image_topic = "/camera/depth/image_raw"
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
#			self.cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
			self.cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

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

	# Image information callback
	def cbInfo(self):

		fontFace = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.5
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

		cv2.putText(
			self.cv_image, "{}".format(self.timestr), 
			(10, 20), 
			fontFace, 
			fontScale, 
			color, 
			thickness, 
			lineType, 
			bottomLeftOrigin
			)

		cv2.putText(
			self.cv_image, "Sample", (10, self.imgHeight-10), 
			fontFace, 
			fontScale, 
			color, 
			thickness, 
			lineType, 
			bottomLeftOrigin
			)

		cv2.putText(
			self.cv_image, "(%d, %d)" % (self.imgWidth, self.imgHeight), 
			(self.imgWidth-100, self.imgHeight-10), 
			fontFace, 
			fontScale, 
			color, 
			thickness, 
			lineType, 
			bottomLeftOrigin
			)

	# Show the output frame
	def cbShowImage(self):
		self.cv_image_clone = imutils.resize(
						self.cv_image.copy(),
						width=320
						)

		cv2.imshow("CameraPreview [Depth]", self.cv_image_clone)
		cv2.waitKey(1)

	# Preview image + info
	def cbPreview(self):
		if self.image_received:
			self.cbInfo()
			self.cbShowImage()
		else:
			rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("CameraPreview [Depth] Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initialize
	rospy.init_node('camera_depth_preview', anonymous=False)
	camera = CameraPreview()
	
#	r = rospy.Rate(10)

	# Camera preview
	while not rospy.is_shutdown():
		camera.cbPreview()
#		r.sleep()
