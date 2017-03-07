
# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output/car_not_car.png "Sample data"
[image2]: ./output/HOG_example.png "HOG visualization"
[image3]: ./output/sliding_windows1.png "Sliding window"
[image4]: ./test_images/test6.jpg "Original image"
[image5]: ./output/test6_hotwindows.jpg "Hot windows"
[image6]: ./output/test6_heatmap.jpg "Heatmap visualization"
[image7]: ./output/test6_labels.jpg "Labels visualization"
[image8]: ./output/test6_final.jpg "Final output"
[image9]: ./output/false_positive.png "False positives elimination"
[image10]: ./output/process_success.png "Vehicle identification"
[video1]: ./output/project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first `Load and Visualize data` sections of the IPython notebook (`P5 - Vehicle Detection - Features Extraction & Training Model`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here is an example using the `RGB` color space and HOG parameters of `channel=0`, `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and based on the accuracy results, final choice of the parameters are chosen (`Selected Settings section` of `P5 - Vehicle Detection - Features Extraction & Training Model` python notebook)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with selected HOG features, as explained in `Save Model with LinearSVC` section of `P5 - Vehicle Detection - Features Extraction & Training Model` python notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with sliding windows implementation for detecting vehicles of different scales - 0.5 (32x32 pixels), 1.0 (64x64 pixels), 1.5 (96x96 pixels) and 2.0 (128x128). In order to speed up and simplify the processing of images, I have decided to drop 0.5 scale sliding windows.

The area that I have chosen for sliding window search was from 400 to 656 pixels in height from the top of the image (out of 720 pixels height). Since there isn't much difference/gain on the processing time on the considered scales (1, 1.5 and 2.0), I haven't optimized further on the sliding window areas for different scales.

![alt text][image3]

Instead of getting HOG features for each of the sliding window, HOG features of the entire sliding window region is extracted and processed for vehicle detection, to be more efficient.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales (1.0, 1.5 and 2.0) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

For training the classifier, 20% samples are used for cross validation. Also, I have utilized stratify option on splitting the samples for training & validation for stratified distribution of vehicles and non-vehicles.

Here is an example image of the pipeline:

![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output/project_video_out.mp4) or, can be accessed online at [here (youtube)](https://youtu.be/TtTJHJcb924).

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

* I recorded the positions of positive detections in each frame of the video.  
* From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
* I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
* I then assumed each blob corresponded to a vehicle.  
* I constructed bounding boxes to cover the area of each blob detected.
* Refer to `pipeline` method in `VehicleDetection` class of the of `P5 - Vehicle Detection - Detect Vehicles` python notebook

##### Original image
![alt text][image4]
##### Hot windows identified, from sliding windows approach
![alt text][image5]
##### Heat map visualization
![alt text][image6]
##### Labels visualization
![alt text][image7]
##### Bounding box, final output
![alt text][image8]

False positives are filtered using thresholding and heatmap history, as shown in the example below:
![alt text][image9]
---

### References

#### 1. [P5 - Vehicle Detection - Features Extraction & Training Model](./P5 - Vehicle Detection - Features Extraction & Training Model.ipynb)
#### 2. [P5 - Vehicle Detection - Detect Vehicles](./P5 - Vehicle Detection - Detect Vehicles.ipynb)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

* One of the biggest challenge for me is the processing time for sliding window approach, it is taking up quite a bit of time. Could explore more to find ways to optimize.
* In the final video, there are a few false positives even after thresholding. This could be reduced/eliminated by having higher thresholding value and increasing more number of hot windows for correct vehicles, more sliding windows/processing.
* Explore more to find other interesting objects like speed limits/traffic signs and integrate with advanced lane lines.
