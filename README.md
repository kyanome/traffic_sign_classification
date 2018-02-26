# Traffic_sign_classification



## ◆ Data Set Summary & Exploration
### 1. Summary of the data set.

- The size of training set is **34799**
- The size of the validation set is **4410**
- The size of test set is **12630**
- The shape of a traffic sign image is **(32, 32, 3)**
- The number of unique classes/labels in the data set is **43**

I used the functions of the below
  1. **len()function** to calculate the size of training set
  2. **shape** to calculate the shape of a traffic sign image
  3. **np.unique()** and **size** to calculate the number of unique labels


### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing distribution of labels.

![alt text][image1]

-------
## ◆ Design and Test a Model Architecture

### 1. How I preprocessed the image data.
##### Gray scale
As a first step, I decided to convert the images to grayscale to be easier for my classifier to learn.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

##### Normalized
As a next step, I normalized the image data. I learned it helps to stabilize the calculattion. ex) increasing speed of training and performance.


This is an result of a original image before and after normalization.

![alt text][image3]


### My final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image							|
| Convolution 5x5    	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6		|
| Convolution 5x5   | Convolution 5x5  |
| Max pooling		| 2x2 stride, outputs 5x5x16      									|
|	Flatten	|	outputs 400			|
|	Fully connected	|	output = 120 |
|		RELU	|												|
|Fully connected |output = 84	|
|	Drop out	|			keep_prob = 	0.8						|
|	RELU	|												|
|	Fully connected	|	output = 43|

#### 1. Describe how you trained your model.
To train the model, I used a LeNet which I learned udacity lecture before to train the model.Details are noted below.
  - optimizer : Adam
  - learning rate : 0.001
  - epochs : 15
  - batch size : 128

#### 2. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of **0.982**
* validation set accuracy of **0.971**
* test set accuracy of **0.886**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * I used a Lenet architecture based on convolutinal neuaral network.I use it because I learned this architecture is good performance.

* What were some problems with the initial architecture?
  * first, I did not undarstand what architecture LeNet is, so I tried to undarstand it through the Looking back at the past of the lecture.
  * Lenet architecture overfitted training data so the accuracy did not rise much

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * I used techniques like dropout to not overfit the data.this was very good to not overfit the data.
* Which parameters were tuned? How were they adjusted and why?
  * I went with the slow learning rate of 0.001 as I wanted my model to be more robust.



---------
### ◆ Test a Model on New Images

#### 1. Five German traffic signs found on the web and provide them in the report.

For each image, I discuss what quality or qualities might be difficult to classify.They where rescaled to 32x32.

Here are five German traffic signs that I found on the web:

![alt text][image4]![alt text][image5]
![alt text][image6]![alt text][image7]
![alt text][image8]


The first image might be difficult to classify because ...

The Last image(No passing) might be difficult to classify because the character in the photograph is written in the image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.


Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield      		| Yield  								|
| No entry   			| No entry 										|
| 50 km/h				| Ahead only											|
| 60 km/h	      		| 60 km/h				 				|
| No passing			| 60 km/h		      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set.

#### 3. The top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.（Analyze Performance）

 The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00       			| Yield   									|
| 1.00     				|  No entry 										|
| 0.51					| Ahead only										|
| 0.99	      			| 60 km/h				 				|
| 0.23				    | 60 km/h		   							|


1. For the first image, the model is relatively sure that this is a Yield (probability of 1.00), and the image does contain a Yield.

2. For the second image, the model is relatively sure that this is a No entry (probability of 1.00), and the image does contain a No entry.

3. For the third image, the model is relatively sure that this is a Ahead only (probability of 0.51), and the image does contain a 50 km/h.

4. For the 4th image, the model is relatively sure that this is a 60 km/h (probability of 0.99), and the image does contain a 60 km/h.

5. For the last image, the model is relatively sure that this is a 60 km/h (probability of 0.23), and the image does contain a No passing.


---------



[//]: # (Image References)

[image1]: ./images/histogram.png "Histogram"
[image2]:./images/img2gray.png "Grayscaling"
[image3]: ./images/normalization.png "Normalization"
[image4]: ./images/17_no_entry.jpg "Traffic Sign 1"
[image5]: ./images/2_50km.jpg "Traffic Sign 2"
[image6]: ./images/3_60km.jpg "Traffic Sign 3"
[image7]: ./images/9_no_passing.jpg "Traffic Sign 4"
[image8]: ./images/13_yield.jpg "Traffic Sign 5"
