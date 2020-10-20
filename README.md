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
