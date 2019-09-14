# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, I have used deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The trained model outputs a steering angle to an autonomous vehicle.

Using a virtual simulator, I drived a car around a track for data collection. The collected image data and steering angles are used to train a neural network and then the model will be used for driving the car autonomously around the track.

The following five files are the important files for the project among others: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car )
* model2.h5 (a trained Keras model)
* a report writeup file (markdown file describing project development process)
* Output_run1.mp4 (a video recording of the vehicle driving autonomously around the track)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
