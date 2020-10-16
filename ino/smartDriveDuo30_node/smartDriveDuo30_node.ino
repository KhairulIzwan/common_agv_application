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
//  Get the ros topic value
  transVelocity = msg.linear.x;
  rotVelocity = msg.angular.z;
  
//  Differential Drive Kinematics
//::http://www.cs.columbia.edu/~allen/F15/NOTES/icckinematics.pdf
//  Differential Drive Kinematics
//::https://snapcraft.io/blog/your-first-robot-the-driver-4-5

//  Step 1: Calculate wheel speeds from Twist
  leftVelocity = transVelocity - ((rotVelocity * wheelSep) / 2);
  rightVelocity = transVelocity + ((rotVelocity * wheelSep) / 2);
  
//  Step 2: Convert wheel speeds into duty cycles
  leftDutyCycle = (255 * leftVelocity) / 0.22;
  rightDutyCycle = (255 * rightVelocity) / 0.22;

//  Ensure DutyCycle is between minimum and maximum
  leftPWM = clipPWM(abs(leftDutyCycle), 0, 25);
  rightPWM = clipPWM(abs(rightDutyCycle), 0, 25);

//  motor directection helper function
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
//  Forward
  if (leftDutyCycle > 0 and rightDutyCycle > 0)
  {
    speedLeft=-leftPWM;
    speedRight=rightPWM;
  }
//  Backward
  else if (leftDutyCycle < 0 and rightDutyCycle < 0)
  {
    speedLeft=leftPWM;
    speedRight=-rightPWM;
  }
//  Left
  else if (leftDutyCycle < 0 and rightDutyCycle > 0)
  {
    speedLeft=leftPWM;
    speedRight=rightPWM;
  }
//  Right
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
