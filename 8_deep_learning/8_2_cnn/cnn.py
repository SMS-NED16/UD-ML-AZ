#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:25:49 2019

@author: saadmashkoor
"""

"""cnn.py - Keras-based Convolutional Neural Network to identify pictures of cats 
and dogs. Assumed dataset with test set/training set of cats/dogs subdirectories
is present ing working directory. Dataset has 8k train, 2k test images (80-20
split, with 50-50 division in each class).
"""


""""------------------------------BUILDING A CNN------------------------------"""
# import required libraries and modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# initialise the CNN
cnn_classifier = Sequential()

# Convolution and ReLU
cnn_classifier.add(
        # Num of filters, rows/cols of filter window, input rows/cols/channels
        # activation function to remove non-linearity and 'negative' pixels
        Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Max Pooling - make spatially invariant, reduce feature map size -> time complexity
cnn_classifier.add(MaxPooling2D(pool_size=(2, 2)))      # halves the feature map size

# Add another convolutional layer
cnn_classifier.add(
        Convolution2D(32, 3, 3, activation='relu'))

# Max Pooling
cnn_classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten - reshape MaxPooling O/P feature maps into vector for ANN input 
cnn_classifier.add(Flatten())

""""------------------------------BUILDING ANN--------------------------------"""
# Hidden layer for the ANN. Input is flattened CNN output.
cnn_classifier.add(Dense(units=128 ,                    # not too small, not too big - tweak this                       
              activation='relu',                
              kernel_initializer='uniform')) 

# Output layer - will predict if input is a dog or not
cnn_classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

""""----------------------------COMPILING NETWORK-----------------------------"""
# Compile the model - prep it for training and prediction of binary classification
cnn_classifier.compile(optimizer='adam', loss='binary_crossentropy',
                       metrics=['accuracy'])

""""----------------------------IMAGE PROCESSING------------------------------"""
"""Data augmentation - creates many batches of our images and increase training
set by transforming/scaling/shifting/rotating our original training set images.
All transformations are random, so helps reduce overfitting on small datasets."""


# Image Augmentation code from https://keras.io/preprocessing/image/
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Generator object defining transformations for training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Data Generator object defining transformations for test set - rescales px only
test_datagen = ImageDataGenerator(rescale=1./255)

# Create training set 
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Create test set
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Apply the generator to the classifier model
cnn_classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,               # images in training set
        epochs=10,
        validation_data=test_set,
        validation_steps=2000/32)              # images in test set