**Advanced Lane Finding Project**

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

[image1]: ./output_images/cam_undistort.png "Undistorted"
[image2]: ./output_images/undistorted.png "Road Undistorted"
[image3]: ./output_images/thresholded.png "Binary Example"
[image4]: ./output_images/warpped.png "Warp Example"
[image5]: ./output_images/lane_fit.png "Fit Visual"
[image6]: ./output_images/overlay.png "Overlayed Image with Lane Highlight"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
#### Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
#### Camera Calibration

##### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is shown from line 8 to 38 in the program file `code.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

#### Pipeline (single images)

##### 1. Provide an example of a distortion-corrected image.
Using the extracted undistortion matrices from the camera calibration module, I can undistort camera images like this test image below:

![alt text][image2]

##### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 43 through 110 in `code.py`).  Here's an example of my output for this step.  (note: this is the same test image as before)

![alt text][image3]

##### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 113 through 118 in the file `code.py`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
src = np.float32(
	[[680, 450],
	 [1120, 720],
	 [200, 720],
	 [590, 450]])

dst = np.float32(
	[[850, 0],
	 [900, 720],
	 [300, 720],
	 [300, 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 680, 450      | 850, 0        | 
| 1120, 720      | 900, 720      |
| 200, 720     | 300, 720      |
| 590, 450      | 300, 0        |

I verified that my perspective transform was working as expected comparing the unwarpped and warpped images as shown below.

![alt text][image4]

##### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I fit my binary lane lines with a 2nd order polynomial kinda like this below (line 218 to 224 of the `code.py`):

![alt text][image5]

##### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 227 through 240 in my code in `code.py`

##### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 244 through 260 and 418 through 429 in my code in `code.py`.  Here is an example of my result on a test image without the additional text annotation:

![alt text][image6]

---

#### Pipeline (video)

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

#### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The most challenging part I observed during the project was robust lane line detection. Under different lightning and road conditions, it could lose track of the detected lines. Applying some moving average filter and tuning line detection window size etc. could help, which is why my pipeline can work relatively robust in the project_video.mp4. But I found it would fail for the challenge videos. More image processing is necessary to make the line detection more robust - I should try adding additional filtering and color enhancement techniques to reduce thresholded background noise. Even deep learning techniques could be utilized to help identify lane lines. Due to the time limit, I didn't explore these options this time.

