# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./output/model_architecture_summary.png "Model Visualization"
[image2]: ./output/center.png "Center driving"
[image3]: ./output/recovery.png "Recovery training"
[image4]: ./output/recover2.jpg "Recovery Image 2"
[image5]: ./output/recover3.jpg "Recovery Image 3"
[image6]: ./output/flipped.png "Flipped Image"
[image7]: ./output/brightness.png "Brightness adjustment"
[image8]: ./output/shadow.png "Shadow effect"
[image9]: ./output/gamma.png "Gamma correction"
[image10]: ./output/trans.png "Translation effect"

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* *model.py* containing the overall script to create and train the model
* *drive.py* for driving the car in autonomous mode
* *model.h5* containing a trained convolution neural network 
* *README.md* summarizing the results
* *utils.py* containing utility functions
* *generator.py* containing data generator class implementation
* *dataloader.py* containing data loader class implementation
* *modelarchitecture.py* containing keras based cnn implementation
* *settings.py* contains a class implementation for command line settings
* *video.py* contains the code for generating video output from the images during autonomous self driving car test runs in the udacity simulator

### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The *modelarchitecture.py* file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is based on **NVIDIA's CNN** (*modelarchitecture.py lines 37-59*) consists of three 5x5 convolution layers with 2x2 strides, two 3x3 convolution layers with 1x1 strides, four fully connected layers ending with a single output.

The input data (image) is normalized in the model using a Keras lambda layer (*modelarchitecture.py line 32*). The input image of size 160x320 pixels, is cropped to 66x200 pixels size to leave out the less useful data (sky, trees and hood of the car).

### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (*modelarchitecture.py lines 43 and 50*).

The model was trained and validated on different data sets for each of the EPOCH to ensure that the model was not overfitting (*generator.py*). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (*modelarchitecture.py line 62*).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Data collection strategies include:

* Three laps of driving in center of the lane
* Two laps of recovery from left and right edges of the lane
* One lap of recovery from left and right edges from the second track

For details about how the training data split/used for training the model, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to start with NVIDIA CNN model. I thought NVIDIA CNN model might be appropriate because it was one of the proven model to have worked with a small amount of training data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers and different training and validation sets for each of the EPOCH.

The final step was to run the simulator to see how well the car was driving around track one with two laps center driving data. There were a few spots where the vehicle fell off the track. To fix the issues, 

* one more lap (3rd lap) of center driving data collected 
* two laps of recovery from left and right edges are collected. 
* To normalize for track 2 driving, one lap of track 2 data is collected/used for training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture (*modelarchitecture.py*) consisted of a convolution neural network with the following layers and layer sizes.

![alt text][image1]

### 3. Creation of the Training Set & Training Process

#### 1. Center driving

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

#### 2. Recovery training

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center of the lane. These images show what a recovery looks like: starting image, recovery in-progress image and end of recovery image.

![alt text][image3]

Then, I repeated this recovery process on track two in order to get more data points.

#### 3. Data augmentation with random effects

**Image Flipping**

To augment the data sat, I also flipped images and angles so that this would reduce/balance left-shifting issues. For example, here is an image that has then been flipped:

![alt text][image6]

**Brightness Adjustment**

To reduce the effect of different weather/light conditions, random adjument of brightness added to augment the data set. For example, here is an image that has been adjusted with brightness:

![alt text][image7]

**Shadow effect**

To reduce the effect on different track conditions with potential shadow effects on the track from objects around the track, random shadow effects are added to the samples. For example, here is an image that has been added with shadow effect:

![alt text][image8]

**Gamma adjustment**

Similar to brightness adjustments, random gamma correction is added to augment the data set. For example, here is an image that has been adjusted with gamma correcion:

![alt text][image9]

**Translation**

Random translation effects are added to the samples to simulate driving in slope conditions. Appropriate adjustments are added to steering angle (0.002).

![alt text][image10]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
