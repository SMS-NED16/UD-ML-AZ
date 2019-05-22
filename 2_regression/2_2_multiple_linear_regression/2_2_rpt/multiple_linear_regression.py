#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:24:02 2019

@author: saadmashkoor
"""

"""Reinforcement of building multiple linear regression models with skleanr and statsmodels"""

"""DATA PREPROCESSING"""
# Import libraries
import numpy as np
import pandas as pd 

# Import dataset
dataset = pd.read_csv('50_Startups.csv')

# Extracting features from dataset