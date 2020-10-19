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
│   ├── face_detection.py
│   ├── facial_landmarks.py
│   ├── object_detection_deep_learning.py
│   ├── pedestrian_detection.py
│   ├── person_detection_deep_learning.py
│   └── teleop_key.py
├── setup.py
└── src
    └── common_agv_application
        ├── centroidtracker.py
        ├── centroidtracker.pyc
        └── __init__.py

```

## Requirement (Setup)
### Hardware
OS: Ubuntu Xenial Xerus (16.04 LTS) or Ubuntu Bionic Beaver (18.04 LTS)
ROS: ROS Kinetic Kame ROS Melodic Morenia
Astra: 

### ROS Package


## smartDriveDuo30_node.ino
Required to download on to the Arduino board

## teleop_key.py
Keyboard-based "AGV" control

## camera_preview.py
Previewing an image stream from camera -- Astra

## facial_landmarks.py


## face_detection.py


## object_detection.py
