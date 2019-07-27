# **Traffic Sign Recognition** 

---

**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visuals/Sample_images.jpg "Sample training image"
[image2]: ./visuals/Class_dist.jpg "Class Distributions"
[image3]: ./test_images/Traffic&#32;signals.jpg "Traffic Signals"
[image4]: ./test_images/Right-of-way&#32;at&#32;the&#32;next&#32;intersection.jpg  'Right-of-way at the next intersection'
[image5]: ./test_images/Priority&#32;road.jpg 'Priority Road'
[image6]: ./test_images/General&#32;caution.jpg 'General Caution'
[image7]: ./test_images/Children&#32;crossing.jpg 'children Crossing'
[image8]: ./visuals/test_images_0.jpg "Priority Road"
[image9]: ./visuals/test_images_1.jpg "Children Crossing"
[image10]: ./visuals/test_images_2.jpg "General Caution"
[image11]: ./visuals/test_images_3.jpg "Right-of-way at the next intersection"
[image12]: ./visuals/test_images_4.jpg "Traffic signals"
[image13]: ./examples/placeholder.png "Traffic Sign 5"
[image14]: ./visuals/Grayscale.png 'Grayscale image'
 
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. A basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799. 
* The size of the validation set is 4410.
* The size of test set is 12620.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43. 

#### 2. Exploratory visualization of the dataset.

9 sample images drawn from the training dataset, are displayed below with corresponding labels as title. Each traffic signs are zoomed in to a square picture with pixel size of 32x32. As image resolution is low, some are hardly recognizable with bare eye. 

![alt text][image1]

Next, we will check the distribution of the each classes over the training, validation and test datasets. With vertical axes representing number of images in each class, first we could see that number of images are distributed unevenly across 43 classes for all 3 datasets. At most, the most abundant class category `Speed limit (50km/h)` contains 10 times as many as the least abundant class category `Speed limit (20km/h)`. Therefore, we can conclude that data is imbalanced. So in the further analysis, it should be considered that the classifier might biased towards the most preminent classes, while underfitting to least preminent ones failing to recignize them.  
On the other hand, it looks like the class distributions across training, validation and test datasets considerably even. In other words, overall datasets splitted evenly into the 3 datasets, so that test results can truely represent the generalization performance of the model. 

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Data Preprocessing
Pixel intensity scaling and conversion to grayscale are implemented as a preprocessing to all datasets prior to training, validation and testing. In order to scale each pixel value within range of 0 to 1, I have decided to divide each data points by 255. This scaling method will speed up the optimization time. 
As for grayscaling, I have implemented in a manual way which is dot product between RGB numpy array and vector `[0.299, 0.587, 0.114]`. Gray scaling also help us to speed up the overall training and inference time since gray scale images are one-dimention less than the RGB images.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image14]


#### 2. Final model architecture (model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected		| outputs 120   								|
| RELU                  |                                               |
| Dropout               | keep_prob 0.8                                 |
| Fully connected       | outputs 84                                    |
| RELU                  |                                               |
| Dropout               | keep_prob 0.8                                 |
| Fully connected       | outputs 43                                    |
| Softmax				|            									|

As we see, its backbone is basically the Lenet architecture with some minor modifications such as activation function and dropout


#### 3. Model Training (Optimizer, the batch size, regularizator, number of epochs and hyperparameters)

To train the model, I used an Adam optimizer with a adaptive learning rate scheduler. The learning rate decays in an exponential manner of `decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)` with an initial rate, decay_rate, decay_steps of 0.001, 0.8, 100000, respectively. Batch was set to 32 throughout the entire 100 epochs of training process. I also applied L2 regularizer to the loss function to avoid any overfitting during the training.

#### 4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 100%.
* validation set accuracy of 96%. 
* test set accuracy of 93%.

I have chosen the classic Lenet architecture from the beginning as my backbone model architecture. Even though it is not that powerful compared to current state-ot-the-art CNN architectures. However, it is powerful enough to process and recognize 32x32 pixel grayscale traffic signs with 93% or higher validation accuracy. As its size is relatively small, it also saves our resource such as memory, training time etc. 
At my first trial without any modifications it was a bit overfitting, training accuracy of 100% and validation accuracy of 93%. To that problem have incorporated L2 regularizer to the loss function and added two Dropout layers to fully connected layers. This method help us to improve the validation accuracy up to 96% while keeping the training accuracy as it was. 

### Test a Model on New Images

#### 1. Five German traffic signs for testing the model performance

Here are five German traffic signs that I found on the web:

<img src="./test_images/Traffic&#32;signals.jpg" width="200" height="200" /> <img src="./test_images/Right-of-way&#32;at&#32;the&#32;next&#32;intersection.jpg" width="200" height="200" style="height: 175px;" /> <img src="./test_images/Priority&#32;road.jpg" width="165" height="165" /> <img src="./test_images/General&#32;caution.jpg" width="170" height="170" style="height :175px;" />  <img src="./test_images/Children crossing.jpg" alt="Children Crossing" title="Children Crossing" width="180" height="180" style="height: 175px;"/>

Predictions for the first and fourth images might be mixed with each other because our model does not take account for color information. To succeffully classify them, the model msut captures shapes within the triange. The last sign might also misclassified as `Keep right` as position and ratio of child and adult inside this sign resembles to the two arrows inside `Keep right` sign.

#### 2. Model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			                        |    Prediction	            		   | 
| :------------------------------------:|:-------------------------------------:| 
| Traffig signs                         | Traffic signs                         |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Priority caution                      | Priority road                         |
| General caution     	                | General caution   	    	    	|
| Children crossing             		| Keep right		         			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. As anticipated, model misclassifies `Children crossing` as `Keep right`. But it was able to identify `General caution` and `Traffic signals` as their groud truths.
This accuracy is inferior to the test set of 93%. I assume that this is due under-representativeness of these 5 test images. 

#### 3. Top 5 Softmax probs for the 5 images

The code for making predictions on my final model is located in the second to the last cell of the Ipython notebook. Here I plot each of 5 images with their top 5 predictions done by the model. 

![alt text][image8]
When the model sees `Children crossing` sign, it assigns 98.6% probability for `Keep right` class and only 1.3% probabily for the true `Children crossing` label. This might be caused by `Keep right` class is overpresented in training data. In fact, the dataset contains 3 times as many `Keep right` as `children crossing`. One natural way to deal with this problem is to augment the dataset with more `Children crossing` traffic signs.

![alt text][image9]
![alt text][image10]
![alt text][image11]

As intuitive as it looks, model predictions for both `General caution` and `Traffic signals` contains one another in the top 5 candidates.
![alt text][image12]





