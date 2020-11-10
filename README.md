# common_agv_application

## About
A project named as **"Automated Guided Vehicle"**

*Computer-controlled and wheel-based, automatic guided vehicles (AGV) are load carriers that travel along the floor of a facility without an onboard operator or driver. Their movement is directed by a combination of software and sensor-based guidance systems.*

## Project structure
```
├── CMakeLists.txt
├── etc
│   └── MDDS30 User's Manual.pdf
├── ino
│   └── smartDriveDuo30_node
│       └── smartDriveDuo30_node.ino
├── model
│   ├── haarcascade_frontalface_default.xml
│   ├── MobileNetSSD_deploy.caffemodel
│   ├── MobileNetSSD_deploy.prototxt.txt
│   └── shape_predictor_68_face_landmarks.dat
├── msg
│   ├── centerID.msg
│   ├── depthID.msg
│   └── personID.msg
├── package.xml
├── README.md
├── script
│   ├── camera_depth_preview.py
│   ├── camera_rgb_preview.py
│   ├── face_detection_haar.py
│   ├── facial_landmarks_dlib.py
│   ├── object_detection_deep_learning.py
│   ├── obstacle_avoidance.py
│   ├── opencv_object_tracker.py
│   ├── person_detection_deep_learning.py
│   ├── person_detection_hog.py
│   └── teleop_key.py
├── setup.py
└── src
    └── common_agv_application
        ├── centroidtracker.py
        ├── centroidtracker.pyc
        ├── __init__.py
        └── trackableobject.py
```

## Requirement (Setup)
### Hardware
1. PC/Laptop: Ubuntu Xenial Xerus (16.04 LTS) -- ROS Kinetic Kame
2. Raspberry Pi 4 8GB: Ubuntu Bionic Beaver (18.04 LTS) -- ROS Melodic Morenia
3. Astra Camera
4. Arduino: MakerUNO
5. Motor Driver: SmartDrive
6. 24V Battery (Lead-ACID)
7. Robot Chassis

<!--**ros_astra_camera packages required** : https://github.com/orbbec/ros_astra_camera.git-->
<!--4. RPLidar-->
<!--**rplidar_ros packages required** : https://github.com/Slamtec/rplidar_ros.git-->

### Software
1. Ubuntu OS:
	1. Download: 
		1. https://releases.ubuntu.com/16.04/
		2. https://releases.ubuntu.com/18.04/
	2. Installation: 
		1. https://ubuntu.com/tutorials/install-ubuntu-desktop-1604#1-overview
		
2. Robot Operating System (ROS)
	1. Kinetic Kame:
		1. http://wiki.ros.org/kinetic/Installation/Ubuntu
	2. Melodic Morena:
		1. http://wiki.ros.org/melodic/Installation/Ubuntu
		
3. Arduino IDE:
	1. https://www.arduino.cc/en/software
	
## System Architecture

## How to use [Terminal]

1. Clone **common_agv_application** package:
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/KhairulIzwan/common_agv_application.git
$ cd ~/catkin_ws && rosdep install -y --from-paths src --ignore-src --rosdistro kinetic && catkin_make && rospack profile
```
2. Clone **ros_astra_camera** package:
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/orbbec/ros_astra_camera.git
$ cd ~/catkin_ws && rosdep install -y --from-paths src --ignore-src --rosdistro kinetic && catkin_make && rospack profile
```
3. Installation **rosserial_arduino** package:
	1. http://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup

## Scripts
## smartDriveDuo30_node.ino
```c++
/*
 * Title: Control MDDS30 in PWM mode with Arduino
 * Author: Khairul Izwan 16-10-2020
 * Description: Control MDDS30 in PWM mode with Arduino
 * Set MDDS30 input mode to 0b10110100
 */

//include necessary library
#include <ros.h>
#include "std_msgs/String.h"
#include <geometry_msgs/Twist.h>

#include <Cytron_SmartDriveDuo.h>
#define IN1 4 // Arduino pin 4 is connected to MDDS30 pin IN1.
#define AN1 6 // Arduino pin 5 is connected to MDDS30 pin AN1.
#define AN2 5 // Arduino pin 6 is connected to MDDS30 pin AN2.
#define IN2 3 // Arduino pin 7 is connected to MDDS30 pin IN2.

Cytron_SmartDriveDuo smartDriveDuo30(PWM_INDEPENDENT, IN1, IN2, AN1, AN2);

//Change according to the robot wheel dimension
#define wheelSep 0.5235 // in unit meter (m)
#define wheelRadius 0.127; // in unit meter (m)

//Variables declaration
float transVelocity;
float rotVelocity;

float leftVelocity;
float rightVelocity;

float leftDutyCycle;
float rightDutyCycle;

float leftPWM;
float rightPWM;

signed int speedLeft, speedRight;

//Callback function for geometry_msgs::Twist
void messageCb_cmd_vel(const geometry_msgs::Twist &msg)
{
	//Get the ros topic value
	transVelocity = msg.linear.x;
	rotVelocity = msg.angular.z;
	
	//Differential Drive Kinematics
	//http://www.cs.columbia.edu/~allen/F15/NOTES/icckinematics.pdf
	//Differential Drive Kinematics
	//https://snapcraft.io/blog/your-first-robot-the-driver-4-5

	//Step 1: Calculate wheel speeds from Twist
	leftVelocity = transVelocity - ((rotVelocity * wheelSep) / 2);
	rightVelocity = transVelocity + ((rotVelocity * wheelSep) / 2);
	  
	//Step 2: Convert wheel speeds into duty cycles
	leftDutyCycle = (255 * leftVelocity) / 0.22;
	rightDutyCycle = (255 * rightVelocity) / 0.22;

	//Ensure DutyCycle is between minimum and maximum
	leftPWM = clipPWM(abs(leftDutyCycle), 0, 25);
	rightPWM = clipPWM(abs(rightDutyCycle), 0, 25);

	//motor directection helper function
	motorDirection();
}

//Helper function to ensure DutyCycle is between minimum
//and maximum
float clipPWM(float PWM, float minPWM, float maxPWM)
{
	if (PWM < minPWM)
	{
		return minPWM;
	}
	else if (PWM > maxPWM)
	{
		return maxPWM;
	}
	return PWM;
}

//Motor Direction helper function
void motorDirection()
{
	//Forward
	if (leftDutyCycle > 0 and rightDutyCycle > 0)
	{
		speedLeft=-leftPWM;
		speedRight=rightPWM;
	}
	//Backward
	else if (leftDutyCycle < 0 and rightDutyCycle < 0)
	{
		speedLeft=leftPWM;
		speedRight=-rightPWM;
	}
	//Left
	else if (leftDutyCycle < 0 and rightDutyCycle > 0)
	{
		speedLeft=leftPWM;
		speedRight=rightPWM;
	}
	//Right
	else if (leftDutyCycle > 0 and rightDutyCycle < 0)
	{
		speedLeft=-leftPWM;
		speedRight=-rightPWM;
	}
	else if (leftDutyCycle == 0 and rightDutyCycle == 0)
	{
		speedLeft=0;
		speedRight=0;
	}
	smartDriveDuo30.control(speedLeft, speedRight);
}

//Set up the ros node (publisher and subscriber)
ros::Subscriber<geometry_msgs::Twist> sub_cmd_vel("/cmd_vel", messageCb_cmd_vel);

ros::NodeHandle nh;

//put your setup code here, to run once:
void setup()
{
	//Initiate ROS-node
	nh.initNode();
	nh.subscribe(sub_cmd_vel);
}

//put your main code here, to run repeatedly:
void loop()
{
	nh.spinOnce();
}
```
- [x] Upload the scripts to Arduino
<!--- [x] Download it first using Arduino IDE -->
<!--**rosserial library required** : http://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup-->

## camera_preview.py
- [x] Previewing an image stream from camera
```python
#!/usr/bin/env python

################################################################################
## {Description}: Previewing an image stream from camera [RGB]
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

		rospy.logwarn("CameraPreview [RGB] Node [ONLINE]...")

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

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

		cv2.imshow("CameraPreview [RGB]", self.cv_image_clone)
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
		rospy.logerr("CameraPreview [RGB] Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initialize
	rospy.init_node('camera_rgb_preview', anonymous=False)
	camera = CameraPreview()
	
#	r = rospy.Rate(10)

	# Camera preview
	while not rospy.is_shutdown():
		camera.cbPreview()
#		r.sleep()
```

<!--## face_detection_haar.py-->
<!--- [x] Detection of face(s) using haar cascade technique-->

<!--## facial_landmarks_dlib.py-->
<!--- [x] Detection of face(s) using dlib libraries-->

## object_detection_deep_learning.py
- [x] Detection of an object using deep learning (MobileNetSSD: https://mc.ai/object-detection-with-ssd-and-mobilenet/)

```python
#!/usr/bin/env python

################################################################################
## {Description}: Object Detection Deep Learning (MobileNetSSD)
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

class ObjectDetection:
	def __init__(self):

		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()

		self.image_received = False

		rospy.logwarn("ObjectDetection Node [ONLINE]...")

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
	def cbObjectDetection(self):
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
				self.box = self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
				(startX, startY, endX, endY) = self.box.astype("int")

				# display the prediction
				self.label = "{}: {:.2f}%".format(self.CLASSES[self.idx], self.confidence * 100)
#				rospy.loginfo("{}".format(self.label))
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

	# Show the output frame
	def cbShowImage(self):
		cv2.imshow("Object Detection", self.image)
		cv2.waitKey(1)

	# Preview image + info
	def cbPreview(self):
		if self.image_received:
			self.cbObjectDetection()
			self.cbInfo()
			self.cbShowImage()
		else:
			rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		rospy.logerr("ObjectDetection Node [OFFLINE]...")
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initialize
	rospy.init_node('object_detection', anonymous=False)
	obj = ObjectDetection()
	
#	r = rospy.Rate(10)

	# ObjectDetection
	while not rospy.is_shutdown():
		obj.cbPreview()
#		r.sleep()
```
<!--## opencv_object_tracker.py-->
<!--- [x] Tracking an object by selecting the ROI-->

## person_detection_deep_learning.py
- [x] Detection of person(s) using deep learning (MobileNetSSD: https://mc.ai/object-detection-with-ssd-and-mobilenet/)

```
Node [/person_detection]
Publications: 
 * /person/ID [common_agv_application/personID]
 * /person/center [common_agv_application/centerID]
 * /person/depth [common_agv_application/depthID]
 * /rosout [rosgraph_msgs/Log]

Subscriptions: 
 * /camera/depth/image_raw [sensor_msgs/Image]
 * /camera/rgb/camera_info [sensor_msgs/CameraInfo]
 * /camera/rgb/image_raw [sensor_msgs/Image]

Services: 
 * /person_detection/get_loggers
 * /person_detection/set_logger_level


contacting node http://192.168.1.69:39971/ ...
Pid: 18412
Connections:
 * topic: /rosout
    * to: /rosout
    * direction: outbound
    * transport: TCPROS
 * topic: /camera/rgb/camera_info
    * to: /camera/camera_nodelet_manager (http://192.168.1.69:46883/)
    * direction: inbound
    * transport: TCPROS
 * topic: /camera/rgb/image_raw
    * to: /camera/camera_nodelet_manager (http://192.168.1.69:46883/)
    * direction: inbound
    * transport: TCPROS
 * topic: /camera/depth/image_raw
    * to: /camera/camera_nodelet_manager (http://192.168.1.69:46883/)
    * direction: inbound
    * transport: TCPROS
```

## person_detection_hog.py
[x] Detection of person(s) using Histogram of Gradient (HOG)

## teleop_key.py
[x] Keyboard-based "AGV" control

