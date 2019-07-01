

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
[image11]:./output_images/Original&#32;image&#32;with&#32;detected&#32;lane&#32;lines.png "Original and final output"  
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

#### 1. Distortion correction

Using the calculated camera calibration matrix and distoption coefficient, I correct road image distotion:
![alt text][image2]

#### 2. Color and gradient thresholding
I used a combination of color and gradient thresholds to generate a binary image using following thresholding methods.

* Get a binary image thresholded by oriental gradients of image using `cv2.Sobel()`:
![alt text][image3]

* Get a binary image thresholded by gradient magnitude of image using `cv2.Sobel()`:
![alt text][image4]

* Get a binary image thresholded by S channel in HSL color space after color space conversion using `cv2.cvtColor()`:
![alt text][image5]

* Get a binary image thresholded by gradient magnitude of image using `cv2.Sobel()`:
![alt text][image6]

* Finally I used combination of above mentioned thresohding techniques to get final binary image for this step:
![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
