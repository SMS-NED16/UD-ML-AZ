#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:42:17 2019

@author: saadmashkoor
"""

"""Final version of data preprocessing template with extra steps such as imputation, 
encoding categorical data are left out. Will only be included on a case-by-case basis."""

# STEP 1 - IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# STEP 2 - IMPORT THE DATASET
datset = pd.read_csv('Data.csv')

# STEP 3 - EXTRACT FEATURES AND TARGET VARIABLES
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# STEP 4 - TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# STEP 5 - FEATURE SCALING (optional - usually handled by the libraries)
"""
from sklearn.preprocessing import StandardScaler
sc_X = standardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""