**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/car_not_car_hog.jpg
[image3]: ./output_images/car_id.png
[image4]: ./output_images/nocar_id.png
[image5]: ./output_images/car_id_heatmap.png
[video1]: ./project_output.mp4

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
#### Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

#### Histogram of Oriented Gradients (HOG)

##### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in in lines 11 through 23 of the file called `train_clf.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car and non-car examples][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(10, 10)` and `cells_per_block=(1, 1)`:


![hog feature example][image2]

##### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and figured a HOG parameter combination of orientation=9, pixels_per_cell=(8, 8) and cells_per_block=(2,2) could give me a very good validation accuracy. The result isn't very sensitive to the HOG parameter variations as I also include color histogram and spatial bins as the features.

##### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG + color histogram + spatial bins features. The code can be found in line 45 to 50 in my "train_clf.py" and line 24 to line 30 in my "code.py" files. Since I am using various types of features, a scaler is also used to normalize the overall feature array as can be found on line 29 to 31 in my "train_clf.py"

#### Sliding Window Search

##### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search only the bottom half of the image to identify cars to reduce computer workload and latency. Hog features were only calculated once for that region but color histogram and spatial bins were calculated window-by-window. I've used 4 window scales to locate vehicles in the bottom-half region. The relevant code can be found in the "search_cars" function in "code.py" and the "find_cars" function in "util.py".

##### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 2 scales (1.2 and 1.5) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images - notice that windows are combined into one with the aid of a heatmap + threshold filtering technique:

Car detected:
![car detected][image3]

No car detected:
![no car detected][image4]

---

#### Video Implementation

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


##### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The relevant code can be found in the "filter_cars" (line 209 to 222) function in "util.py".
Below shows the corresponding heatmap before and after thresholding for the car detected case:

![car heatmap][image5]

---

#### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In general the pipleline works pretty well - it can identify cars pretty accurately. However, I notice when road/lighting conditions change, there's a chance the car could lose track. Since car's movement won't be drastically large in a short period of time, applying some additional filtering/tracking should help improve the tracking accuracy. In particular, I can record the bounding box variations over adjacent video frames and discard/smooth those sudden changes. Also the classifier training set could be further enhanced to include images under different lighting conditions. These extra steps should help improve the overall tracking accuracy further more.
