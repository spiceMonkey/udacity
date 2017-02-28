#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/spiceMonkey/udacity/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the Numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (width x height x channel)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 
Here is a randomly sampled traffic sign image from the training set.

![Random Training Image](/writeup_images/rndm_img_train.png)

And here is a bar chart showing how the training/validation/testing data is distributed in the data set. We can clearly see there are more data for some classes over the others.

![Old Data Dist.](/writeup_images/old_dataset_stat.png)


###3.Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Knowing that there are more data for some classes than others which could cause biases in the final classificaiton accuracy, I decide to augment the original dataset by creating more artifical ones.

Code cell 4 rebundle the original training, validation and testing data to ease later image creations. And below is the overall statistics overall different classes, which once again confirmed the bias.

![Old Data All](/writeup_images/old_tot_stat.png)

I implemented up to 5 image transformation methods as shown in code cell 5. They are Gaussian blurs, 30 degree CCW rotation , 30 degree CW rotation, dilation and erosion. Image below shows an example how these effects are manifastied on the original image.

![Img Aug](/writeup_images/img_aug.png)

Code cell 6 shows how the image augmentation techniques are applied on the original dataset. The code is self-explanatory and depending on how many images each class already has, I randomly select some images from each class and apply a varying number of transformations on these images to create new ones. New data are saved to pickle files and thus this section can be bypassed for later runs.

The artificial data are then combines with the original dataset to create a new dataset in code cell 7. After shuffling, the new set is then splitted again into training, validation and testing set with the same splitting ratio as the original. The figure below shows the new statistics for each set.

![New Data Dist.](/writeup_images/new_dataset_stat.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 8th code cell of the IPython notebook.

As a first step, I decided to do a max-min normalization on each of the 3 RGB channels of the dataset. This is to help reduce image variations such as from different light conditions. 

Here is an example of a traffic sign image before and after the normalization. We can clearly see that the image quality is enhanced - for this case, a darker image becomes more visible.

![Image Process](/writeup_images/img_proc.png)


####2. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 9th cell of the iPython notebook. 

My final model consisted of the following layers:

| Layer         		|          Description	        			                		| 
|:---------------:|:-------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							                       | 
| Convolution 3x3 | 1x1 stride, VALID padding, outputs 30x30x32 	     |
| RELU					       |		                                                 |
| Convolution 3x3 | 1x1 stride, VALID padding, outputs 28x28x32      	|
| RELU					       |	                                                  |
| Max pooling	    | 2x2 max, 1x1 stride (overlap pooling), outputs 27x27x32 	|
| Random Dropout  | 0.5 probability                                   |
| Convolution 3x3 | 1x1 stride, VALID padding, outputs 25x25x64 	     |
| RELU					       |                                                   |		   
| Convolution 3x3 | 1x1 stride, VALID padding, outputs 23x23x64 	     |
| RELU					       |	                                                  |
| Max pooling	    | 2x2 max, 1x1 stride (overlap pooling),  outputs 22x22x64 |
| Random Dropout  | 0.5 probability                                   |
| Fully connected	| input 22x22x64, output 512                        |
| RELU					       |                                                   |		
| Random Dropout  | 0.5 probability                                   |
| Fully connected	| input 512, output 43                              |
| Softmax				     |                                                   |



####3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 10th to 14th cell of the ipython notebook. 

To train the model, I used a batch size of 128 and epoch number of 15. I used Adam optimizer to take the advantage of adaptive learning rate. Initial learning rate is set to be 0.001.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.957
* test set accuracy of 0.959

Note that I had the following iterations before settling down to the above CNN architecture. 
If an iterative approach was chosen:
* I used with the LeNet-5 architure from the previous lab assignment as a starting point, however I found its accuracy is to lower than 90%. 
* The LeNet-5 architecture is relatively shallow and has a pretty small layer depth. Also, each conv layer is directly followed by a destructive max pool layer which I believe is not good for images with rich features such as these traffic signs. 
* For that matter, I came up with the above architecture which had two consecutive conv layers before a max layer and had an progressively increase depth.
* I also notice that lower the sigma value in the truncated Gaussian randomalization for the weights coefficent can improve the final accuracy. I believe this is because the "optimal" coefficients are actually very close to 0. So having them randomalized closer to 0 is easier for the gradient descent to reach minimum.
* During the iteration, I also found adding dropout layers are quite helpful in avoiding overfitting. In fact, I got a very close resut between the validation accuracy and test accuracy, indicating the model is not overfitted.
* I also tried a larger CNN - e.g. adding one more set of conv->conv->pool to above architecture. The return is diminishing and I found the memory requirement is bigger and inference speed is much slower, making large network less attractive for real-time applications that requires low latency.
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Code cell 15 loads the images I randomly downloaded from the web. The functions also resize the images and apply the same processing method to them.

Below are the 5 images:

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------:|:---------------------------:| 
| 50 km/h      	  | 50 km/h   									         | 
| General Caution | General Caution 										|
| Pedestrians				 | Pedestrians											   |
| Road work      		| Road work					 				|
| Roundabout			| Roundabout      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 10%. This compares favorably to the accuracy on the test set of 96%.

Notice that the images I found are relatively clear so achieving such a high accuray isn't suprising to me. I was originally thinking the "general caution" and "pedestrians" images might not be able to detect correctly as they share some similarities. However, both are classified correctly.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For all the 5 images, the model is all pretty sure about that they are. As can be found from the texts that summarized the top 5 class probability - all get almost 100% probability.

![pred1](/writeup_images/pred1.png)

![pred2](/writeup_images/pred2.png)

![pred3](/writeup_images/pred3.png)

![pred4](/writeup_images/pred4.png)

![pred5](/writeup_images/pred5.png)


