# Convolutional Neural Networks - Introduction

## Introduction to Vision
- Our brain looks for features in images/our field of view.
- The way we look at an image changes the features our brain focuses, and in turn changes the way we perceive or understand them.
- Examples: 3 optical illusions that illustrate how the features our brain picks up will help us classify what we're seeing. 
- The way computers process images is very similar to the way computers process them: the emphasis is always on specific features.
- Geoffrey Hinton's neural network outputs **probabilities** of a particular image belonging to a specific category. 

## What are CNNs?
- CNNs are actually overtaking ANNs in terms of popularity!
- It's a very important field that is relevant to emerging technologies such as
	- self-driving cars
	- automatic photo tagging on Facebook
- Popularized by Yann LeCun. Pioneer of AI at FB, a student of Geoffrey Hinton, and a professor at NYU. 
- Input image > CNN > Output label (image class) after training it using labeled/categorized data.
- Every image is an array of pixels. CNNs will treat a black and white array as a 2D array with each pixel having a pixel intensity between 0 and 255 (0 being white, 255 being completely black).
- So the digital form of a picture/image is the foundation of CNN and computer vision.
- Coloured pictures are similar: also an array of pixels but each pixel has 3 channels (RGB) indicating the intensity of each colour. So is a 3d array.


## CNN Workflow
1. Convolution
2. Max Pooling
3. Flattening
4. Full Connection