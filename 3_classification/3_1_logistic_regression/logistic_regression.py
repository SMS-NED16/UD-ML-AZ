#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 08:41:51 2019

@author: saadmashkoor
"""

"""logistic_regression.py - Implementing Logistic Regression for classification
of a Social Network ad data."""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')