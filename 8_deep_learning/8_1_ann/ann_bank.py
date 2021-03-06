#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:57:00 2019

@author: saadmashkoor

ann_bank.py - Using Keras-based Aritifical Neural Network to predict churn rate
for bank customers.
"""

"""----------------------------DATA PREPROCESSING-----------------------------"""
# import libraries
import pandas as pd                         # will read CSV into a dataframe

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# extracting features and labels 
X = dataset.iloc[:, 3:-1].values            # all except index, row num, id, name
y = dataset.iloc[:, -1].values              # did customer leave bank?

# encode categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# create separate label encoder for each categorical variable
labelEncoder_country, labelEncoder_gender = LabelEncoder(), LabelEncoder()

# Fitting encoders
X[:, 1] = labelEncoder_country.fit_transform(X[:, 1])   # country
X[:, 2] = labelEncoder_gender.fit_transform(X[:, 2])    # gender

# One Hot encoding only for the country
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray() 

# Avoid dummy variable trap by dropping one dummy variable for country
X = X[:, 1:]

# splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling is necessary for ANNs
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""-------------------------------CREATING ANN-----------------------------------"""
# Import the Keras library and required packages
# Didn't install keras, used Keras as packaged with TF, so explicit import
from tensorflow.keras.models import Sequential          # sequentially add layers to ANN
from tensorflow.keras.layers import Dense               # densely connected layers

# Initialize the ANN as a sequence of layers
ann_classifier = Sequential()                           # ANN Model

# Add layers to the ANN. Because sequential, first layer obj = first layer added
# Rule of thumb - num of nodes in a layer = avg(nodes in I/P layer, nodes in O/P layer)
# Better approach - parameter tuning - cross validation
# This is hidden layer. Input added implicitly.
ann_classifier.add(
        Dense(units=6,                          # num of units in layer = avg of I/O layer units
              activation='relu',                # rectified linear unit best for hidden
              input_dim=11,                     # 11 features so 11 inputs
              kernel_initializer='uniform'))    # params random init from uniform dist

# Create a second hidden layer - no need to specify input count b/c knows O/Ps of prev layer
ann_classifier.add(
        Dense(units=6,                          
              activation='relu',                
              kernel_initializer='uniform'))    

# Create output layer
ann_classifier.add(
        Dense(units=1,                          # only one output (Y/N)
              activation='sigmoid',             # output layer so sigmoid activation function
              kernel_initializer='uniform',     # weights still init randomly
                ))

"""-------------------------------COMPILING ANN---------------------------------"""
# Optimise the weights of the ANN using gradient descent
ann_classifier.compile(
        optimizer='adam',                       # popular variant of SGD
        loss="binary_crossentropy",             # logarithmic loss for bin classification                          
        metrics=['accuracy'])


"""-------------------------------FIT ANN TO SET---------------------------------"""
# So far the ANN has just been instantiated/set up for training. We now need
# to fit it or optimise it parameters
ann_classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


"""----------------------------------TEST ANN------------------------------------"""
# Make predictions on the test set
y_pred = ann_classifier.predict(X_test)                 # continuous probabilities

# convert continuous predicted probabilities into Y/N binary classes using threshold
y_pred = (y_pred > 0.5)                                 # 1/0 now

# evaluae performance
from sklearn.metrics import classification_report, confusion_matrix
class_report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

"""-----------------------------INTERPRETATION----------------------------------"""
#Got test set accuracy of 84.3%, same as the final accuracy in the training set.