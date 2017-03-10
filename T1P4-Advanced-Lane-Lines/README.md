# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output/calibration.png "Camera calibration"
[image2]: ./output/undistort.png "Undistort example"
[image3]: ./output/unwarped.png "Perspective transform"
[image4]: ./output/sobelx.png "Gradient thresholding"
[image5]: ./output/yellow_white_mask.png "Color thresholding"
[image6]: ./output/pipeline.png "Pipeline example"
[image7]: ./output/pipeline_histogram.png "Histogram"
[image8]: ./output/sliding_window.png "Sliding window and polyfit"
[image9]: ./output/overlay.png "Overlay, Final output"
[video1]: ./output/project_video_out.mp4 "Video"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `Camera Calibration` section of the IPython notebook `P4 - Data Analysis`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

##### Camera calibration
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

##### Distortion correction example
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Refer to `section 3` of the IPython notebook `P4 - Data Analysis` for the implementation of perspective transform.  I chose the hardcode the source and destination points in the following manner:

```
# source points
top_left = (588, 454)
top_right = (692, 454)
bottom_left = (203, 720)
bottom_right = (1105, 720)

src = np.float32([top_right,
                  bottom_right, 
                  bottom_left, 
                  top_left])

# desired destination
dst_top_left = (200, 0)
dst_top_right = (1000, 0)
dst_bottom_right = (1000, 720)
dst_bottom_left = (200, 720)

dst = np.float32([dst_top_right,
                  dst_bottom_right, 
                  dst_bottom_left, 
                  dst_top_left])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Based on various color and gradient transformations on test images (refer to `section 4` and `section 5` in the IPython notebook `P4 - Data Analysis`), identified a combination of the SobelX gradient, Yellow and White color transformations for the pipeline (refer to `section 6` in the IPython notebook `P4 - Data Analysis`)

##### Gradient thresholding example
![alt text][image4]

##### Yellow and White color masking example
![alt text][image5]

##### Pipeline example
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Using histogram of the pipeline image and sliding window approach (refer to section 8 in the IPython notebook `P4 - Data Analysis`), I was able to identify and fit my lane lines with a 2nd order polynomial.

##### Histogram of the pipeline image
![alt text][image7]

##### Sliding window and 2nd order polyfit
![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Refer to `sections 8.3 - 8.5` in the IPython notebook `P4 - Data Analysis` for the calculation of radius curvature and the position of the vehicle with respect to the center of the lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Refer to `section 8.7` in the IPython notebook `P4 - Data Analysis`.  Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

##### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output/project_video_out.mp4)

---

### References

#### 1. [P4 - Data Analysis](./P4 - Data Analysis.ipynb)
#### 2. [P4 - Project](./P4 - Project.ipynb)

---
### Discussion

##### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* One of the biggest challenge was to come up with color/gradient tranform thresholds for different road conditions (shadows, contrast and poor visibility of the lane).
* Pipeline is not robust enough for challenging road conditions like highly curvy lanes since the implementation utlizes only 2nd order poly fit.
* It would be interesting to explore if higher order poly fit could be tried to handle more challenging road conditions.