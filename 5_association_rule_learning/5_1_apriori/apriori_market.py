#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 05:34:24 2019

@author: saadmashkoor
"""

"""apriori.py - Python implementation of the Apriori association rule learning
algorithm for Market Basket Optimisation using Apyori library from the Python
Software Foundation"""

"""
BUSINESS PROBLEM
- Optimising the sales for a grocery store in Southern France.
- Association rule learning can help stores optimise their sales.
- They can use ARL to decide where to place their products in the store.
- By placing closely associated items in a store closer to each other, stores
are more likely to increase sales because customers will purchase both.
- If both A and B are associated, placement of products can prompt the customer
to buy both A and B instead of just one product.
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset - first row is not column names
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

"""
Every entry in the dataset is a summary of all the items purhcased by a 
customer over the course of a week. There are 7501 such observations in the
dataset.

The apriori implementation from PSF library expects the dataset to be a 
list of lists. Need to cast dataframe to list of lists.
"""

# Cast dataset to a list of lists that is compatible with PSF apriori algo
transactions = dataset.applymap(str).values.tolist()

# Train the apriori model using the dataset and the Apyori library
# Choosing values for support, confidence, lift, and min length requires
# extensive trial and error. 
from apyori import apriori

support = 3 * 7 / len(transactions) # product purchased at least thrice per day for all transaction
confidence = 0.20                   # if too high, won't be able to find products
lift = 3                            # association must produce a 30% increase in probability of purchase
min_prods_recommended = 2           # at least two items must be recommended

rules = apriori(transactions,
                min_support=support, min_confidence=confidence, 
                min_lift=lift, min_length=min_prods_recommended)

# summarise rules learnt by the apriori algorithm
# not explicitly sorted by lift because Apyori has its own relevance criteria
# This creates a list of Apyori record objects. 
results = list(rules)  # this will take time

# Translate to readable results
results_list = []
for i in range(len(results)):
    results_list.append("Rule:\t" + str(results[i][0]) + 
                        "\nSupport:\t" + str(results[i][1]) + 
                        "\nConfidence:\t" + str(results[i][2][0][2]) + 
                        "\nLift:\t" + str(results[i][2][0][3]))