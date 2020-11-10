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
- [x] Motor drive script
- [x] Download it first using Arduino IDE 
<!--**rosserial library required** : http://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup-->

## camera_preview.py
[x] Previewing an image stream from camera

## face_detection_haar.py
[x] Detection of face(s) using haar cascade technique

## facial_landmarks_dlib.py
[x] Detection of face(s) using dlib libraries

## object_detection_deep_learning.py
[x] Detection of an object using deep learning (MobileNetSSD)

## opencv_object_tracker.py
[x] Tracking an object by selecting the ROI

## person_detection_deep_learning.py
[x] Detection of person(s) using deep learning (MobileNetSSD)

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

