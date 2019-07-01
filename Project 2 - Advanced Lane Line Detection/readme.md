

## Advanced Lane Finding 
### Overview
In this project, by utilizing traditional computer vision techniques, I developed a software pipeline which detects lane line boundaries on the road. 
Complete jupyter notebook of the pipeline can be found [here](https://github.com/Chimdee/Self-Driving-Car/blob/master/Project%202%20-%20Advanced%20Lane%20Line%20Detection/Advanced%20Lane%20Finding.ipynb). 

![alt text][image0]

### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]:./output_images/final&#32;output.png "Final output"  
[image1]:./output_images/Undistorded&#32;chessboard.png "Undistorted Chessboard"  
[image2]:./output_images/Undistorted&#32;image.png "Undistorted road"  
[image3]:./output_images/Sobel&#32;gradients.png "Sobel Gradients"  
[image4]:./output_images/Sobel&#32;gradients&#32;magnitude.png "Sobel Gradients Magnitude"  
[image5]:./output_images/Sobel&#32;gradients&#32;direction.png "Sobel Gradients Direction"  
[image6]:./output_images/S&#32;channel&#32;in&#32;HSL&#32;color&#32;space.png "S channel thresholded"  
[image7]:./output_images/Thresholded&#32;binary&#32;image.png "Thresholding combined"  
[image8]:./output_images/Perspective&#32;transformed&#32;binary&#32;image.png "Warped image"  
[image9]:./output_images/Warped&#32;image&#32;with&#32;detected&#32;lane&#32;lines.png "Lane Detection"  
[image10]:./output_images/Warped&#32;image&#32;with&#32;detected&#32;lane&#32;lines&#32;(2).png "Lane Detection (2)"  
[image11]:./output_images/Original&#32;image&#32;with&#32;detected&#32;lane&#32;lines.png "Warped back to original image"  
[image12]:./output_images/histogram.png "Pixel histogram"
[video1]:./project_video_output.mp4 "Video"

### Main libraries and dependencies for the project
* [NumPy](www.numpy.org) - A widely used scientific computing library for python
* [OpenCV](www.opencv.org) - A well-known open course computer vision library
* [Matplotlib](www.matplotlib.org) - A plotting library for python 
* [MoviePy](https://zulko.github.io/moviepy/) - Video editing library for python


### Here I will consider each steps individually and describe how I addressed it in my implementation.  

---

### Camera Calibration  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard image using the `cv2.undistort()` function and obtained this result: 

![Chessboard][image1]

#### Distortion correction

Using the calculated camera calibration matrix and distoption coefficient, I correct road image distotion:
![alt text][image2]

#### Color and gradient thresholding
I used a combination of color and gradient thresholds to generate a binary image using following thresholding methods.

* Get a binary image thresholded by oriental gradients of image using `cv2.Sobel()`:

![alt text][image3]


* Get a binary image thresholded by gradient magnitude of image using `cv2.Sobel()`:

![alt text][image4]


* Get a binary image thresholded by gradient direction of image using `cv2.Sobel()` and `np.arctan2()`:

![alt text][image5]


* Get a binary image thresholded by S channel in HSL color space after color space conversion using `cv2.cvtColor()`:

![alt text][image6]


* Finally I used combination of above mentioned thresohding techniques to get final binary image for this step:

![alt text][image7]


#### Perspective Transform
Fisrt, I located source points `src` in the sample image and destination points `dst` in _birds-eye_ as shown below. Then I did prespective transform using  Opencv's `cv2.getPerspectiveTransform()` function. In addition, inverse perpective transform _Minv_ is  calculated here to unwarp sample images later in the pipeline. 

```python
top_left = np.array([560, 460]).reshape(1, -1)
top_right = np.array([730, 455]).reshape(1, -1)
bottom_left = np.array([200, 719]).reshape(1, -1)
bottom_right = np.array([1200, 719]).reshape(1, -1)

src = np.float32([top_left, top_right, 
                 bottom_right, bottom_left])
offset = 100
dst = np.float32([[bottom_left[0][0] - offset, 0], 
                  [bottom_right[0][0] + offset, 0], 
                  [bottom_right[0][0], bottom_right[0][1]], 
                  [bottom_left[0][0], bottom_left[0][1]]]) 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 460      | 100, 0        | 
| 730, 455      | 1300, 0      |
| 1200, 719     | 1200, 719      |
| 200, 719      | 200, 0 719      |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]

#### Lane boundary detection

First, I calculate pixel histogram of bottom half of the warped image to find lane position. Two peaks of the hisgram should be left and right lane positions on the image. 

![alt text][image12]

I will slide a fixed-sized window over the image to find lane line, where histogram value takes its maximum. After that we will fit a second polinomial (`np.polyfit()`)  for both left and right lanes to get complete curves /yellow curves in the below picture/. 

![alt text][image9]

Once I=we have a fitted polynomial, I will use its coefficients for detecting lanes for the next frame. In other words, once I knew where the lanes were in the last frame, I can search around that lanes for the next frames. It prevents us from implementing the sliding window method on entire image for every new video frames coming.
![alt text][image10]


####  Curvature of lane and vehicle offset position
We can calculate radius of road curvatures using a formula in [this](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) tutorial. Also in this step, we have convert our metrics into meters from pixels. Here I am assuming that the lane is about 30 meters long and 3.7 meters wide, and our camera image has 720 relevant pixels in the y-dimension (remember, our image is perspective-transformed!), and we'll say roughly 700 relevant pixels in the x-dimension. Therefore, we can use following multiflier for the conversion. 
`# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension `
When we calculate vehicle offset position from center off the road, we are assuming that the camera is fixed at center of the car front. Therefore along with the detected left and right lane positions, it is easy to calculate the car offset metrics. Namely, we compare middle point of the lane lines with the middle point of image captured by front camera. If the two overlap, it means offset equals to zero implying that vehicle is driving exatcly center of the road. 


#### Warp the image back to the original image
Lanes will be depicted on a blank warped image using `cv2.polyfill()`. Then will map this warped image back to original image using inverse perspective transform matrix _Minv_. With curvature and offset metrics are displayd, final output of our pipeline yields following frame.

![alt text][image11]

---

### Pipeline (test video)

#### Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion
This pipeline is robust to lane curvature, lane line colors and a few lightening conditions, which are challenging problems for basic tradiational computer vision techniques such as Hough transform. But at the same time, there are still some drawbacks in this pipelines which can be improved. For example, choosing hyperparameters for gradient/color thresholding is somewhat handwavy. One value can be optimal for one road condition but could be suboptinal for many others making it very hard to find an universal optimum value. For example, if you watched video output closely, you may noticed that the pipeline jitters once when the camera sees tree shadow on the road. However, it still could work most of the time. I think one way to improve and make it more robust can be testing the parameters in as many environments as possible and fine-tune them.

