# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of my own driving behavior 
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/nvidia_architecture.PNG "Model Visualization"
[image2]: ./examples/center_2019_07_11_11_24_19_955.jpg "center"
[image3]: ./examples/center_2019_07_12_11_36_12_707.jpg "right recovering"
[image4]: ./examples/right_2019_07_11_11_26_48_183.jpg "right"
[image5]: ./examples/center_2019_07_12_11_57_17_639.jpg "track 2"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model2.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 snd 5x5 filter sizes and depths between 32 and 128 (model.py lines 73-87).

The model includes RELU layers to introduce nonlinearity (code line e.g., 77), and the data is normalized in the model using a Keras lambda layer (code line 76). 

#### 2. Attempts to reduce overfitting in the model

Image data is horizontally flipped (code line 57-62) in order to reduce overfitting (model.py lines 21). Also, driving data of courter clock wise driving and driving data from track 2 (code line 32- 37) are collected to prevent the model overfitting to track 1.  

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 66). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 90).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I thought the above-mentioned model might be appropriate because it has proven to be efficient in the literature. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I have collected more data for counter-clock wise driving.

When using the simulator, there were still a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I again collected data of recovering from the road edges. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture of the model is visualized below. It is basically the model architecture which is proposed by NVidia.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture robudt driving behavior of mine, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it fell off from the track For example this image shows what a recovery from the right side looks like:

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also collected images from track 2. For example, here is an image that has then been taken from the track 2:

![alt text][image5]


After the collection process, I had around 42200 number of data points. I then preprocessed this data by cropping out the lower part/exclusion of sky and mountains/ of the images. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20. I used an adam optimizer so that manually training the learning rate wasn't necessary.
