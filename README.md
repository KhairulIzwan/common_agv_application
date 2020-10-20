# common_agv_application

## About
A project named as **"Malaysian Automated Guided Vehicle""**
**Computer-controlled and wheel-based, automatic guided vehicles (AGV) are** 
**load carriers that travel along the floor of a facility without an onboard** 
**operator or driver. Their movement is directed by a combination of software**
**and sensor-based guidance systems.**

## Project structure
```
├── CMakeLists.txt
├── ino
│   └── smartDriveDuo30_node
│       └── smartDriveDuo30_node.ino
├── model
│   ├── haarcascade_frontalface_default.xml
│   ├── MobileNetSSD_deploy.caffemodel
│   ├── MobileNetSSD_deploy.prototxt.txt
│   └── shape_predictor_68_face_landmarks.dat
├── package.xml
├── README.md
├── script
│   ├── camera_preview.py
│   ├── face_detection_haar.py
│   ├── facial_landmarks_dlib.py
│   ├── object_detection_deep_learning.py
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
**ros_astra_camera packages required** : https://github.com/orbbec/ros_astra_camera.git
4. RPLidar
**rplidar_ros packages required** : https://github.com/Slamtec/rplidar_ros.git
5. Arduino: MakerUNO
6. Motor Driver: SmartDrive

## System Architecture


## Scripts
## smartDriveDuo30_node.ino
[x] Motor drive script
[x] Download it first using Arduino IDE 
**rosserial library required** : http://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup

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

## person_detection_hog.py
[x] Detection of person(s) using Histogram of Gradient (HOG)

## teleop_key.py
[x] Keyboard-based "AGV" control


# CameraInfo
header: 
  seq: 156
  stamp: 
    secs: 1603164894
    nsecs:  14866344
  frame_id: "camera_rgb_optical_frame"
height: 480
width: 640
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [519.761474609375, 0.0, 327.35552978515625, 0.0, 519.761474609375, 236.2485809326172, 0.0, 0.0, 1.0]
R: [0.9999855756759644, 0.002107520354911685, -0.004944569896906614, -0.002123440383002162, 0.9999925494194031, -0.003216687822714448, 0.004937754012644291, 0.0032271407544612885, 0.9999825954437256]
P: [519.761474609375, 0.0, 327.35552978515625, -25.509384155273438, 0.0, 519.761474609375, 236.2485809326172, -0.0034834917169064283, 0.0, 0.0, 1.0, -0.19314756989479065]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
