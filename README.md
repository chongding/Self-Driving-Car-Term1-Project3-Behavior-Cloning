# Self-Driving-Car-Term1-Project3-Behavior-Cloning
1.	Model Architecture
I started with the Nvidia network, with all the data augmentations (more details later) and parameter tuning (more details later), I got a working model that can drive the car through the track1. The problem however is that the model parameter file is very large, approximating 90Mbs. So I spent a lot of time on simplifing the model. the final version is about 17Mb. Main difference from the Nvidia model:
  -Add lambda to normalize the input image
  -Add cropping to remove pixles that not part of the driving track
  -Reduce the fully connected layers from 1164 to 200, removed one fully connected layer (100) 
  
Layer         		                 Description	        					
|:-------------------------:|:-------------------------------------------------------------:| 
| Input         		| 160x320x3 RBG image   	
|:-------------------------:|:-------------------------------------------------------------:| 
| Lambda        		| Normalization
|:-------------------------:|:-------------------------------------------------------------:| 
| Cropping      		| 80x320x3 RBG image (top 60 and bot 20 removed)
|:-------------------------:|:-------------------------------------------------------------:| 		
| Convolution 5x5x24   	| 2x2 stride, Valid padding
| RELU											
|:-------------------------:|:-------------------------------------------------------------:| 
| Convolution 5x5x32        | 2x2 stride, Valid padding
| RELU											
|:-------------------------:|:-------------------------------------------------------------:|
| Convolution 5x5x48	| 2x2 stride, Valid padding
| RELU
|:-------------------------:|:-------------------------------------------------------------:|
| Convolution 3x3x64	| 1x1 stride, Valid padding
| RELU
|:-------------------------:|:-------------------------------------------------------------:|
| Convolution 3x3x64	| 1x1 stride, Valid padding
| RELU											
|:-------------------------:|:-------------------------------------------------------------:|
| Flatten						
|:-------------------------:|:-------------------------------------------------------------:|
| Dense		             | output 200    	
| Dropout				
|:-------------------------:|:-------------------------------------------------------------:|
| Dense		             | output 50    	
| Dropout				
|:-------------------------:|:-------------------------------------------------------------:|
| Dense		             | output 10   	
|:-------------------------:|:-------------------------------------------------------------:|
| Output			| outputs 1

2.	Data Augmentation
There are three images (left, center, right) for each steering angle.  In order to use the left and right image, I added 0.09 to the left image and subtracted 0.09 to the right image just to help recover the car to the middle line. So a total of 24108 images from the original training dataset.
Next, I flipped every image to double the training dataset. In the training dataset, the car drive counter-clock wise around the track, so the image was left-turn biased. Flipping will generate right-turn images that helps on generalize the steering.
As mentioned in model architecture section, the input RGB data was first normalized then cropped to focus on the driving track itself.
3.	Attempts to Reduce Over-fitting
Dropout was the main thing, but still see some over-fitting. Also tried add L2 regularization, not much difference. I ended up train less number of epochs before over-fitting happen. 
4.	Model Parameter Tuning
Adam optimizer to 0.0001
dropout keep_prob: 0.4.  
epochs : 3
