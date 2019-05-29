#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 07:49:54 2019

@author: saadmashkoor
"""

"""upper_confidence_bound.py - Reinforcement Learning using UCB to solve the 
multi-armed bandit problem in the context of ads and clickthrough rates"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

"""
--------------------------------Business Problem---------------------------------
An SUV company has prepared 10 different versions of an
advertisement for their vehicle. We want to choose the version of the ad that will
maximize conversion rate i.e. maximize the number of users who click on the ad.
We need to implement a strategy to quickly find out which version of the ad will give
the most clicks.


-----------------------------------Procedure-------------------------------------
We have 10 versions of the ad. Each time a user logs in to the website, we will show
them one version of the ad. Every time the user clicks on the ad, our algo gets a 
reward of 1. Every ad that isn't clicked is a penalty - no reward.

The ad chosen for a given iteration will be based on the results of previous iterations.
It won't be random selection. The model will iteratively learn which ads most users
are likely to click on. If we used random selections, the reward is in the range of 
1100 - 1300. This means the random selection's click conversion rate is only 13%.
We can do better than this!

Also, we get a nearly uniform distribution of the ad click frequency. We don't have
any insights about choosing the best ad.

The dataset is just used for simulation. We won't be using it in the conventional
sense of extracting features/labels/etc. We've created 10k users and predefined
the ads they will click on. This helps us train the network without requiring actual
real time responses from people IRL. 

A user can click on no ads or multiple ads. 
"""

"""
------------------------------UCB Implementation---------------------------------
"""
N_ROUNDS = 10000                    # number of times we will show a version of the ad
d = len(dataset.columns)            # number of ads that can be shown to a user

# List to store number of times each ad has been selected by a user - init 0
numbers_of_selections = [0] * d

# Sum of rewards for each ad - init 0 
sums_of_rewards = [0] * d

# empty list to keep track of ads selected in each round
ads_selected = []

# variable to store the total reward at the end of all iterations
total_reward = 0

# For each round in the `N_ROUNDS` of showing an ad to a user 
for n in range(0, N_ROUNDS):
    
    # create a max upper bound variable for the best ad for each user
    max_upper_bound = 0
    
    # create a variable to remember index of the ad selected
    ucb_ad_index = 0 
    
    # for every ad in the list of possible ads
    for i in range(0, d):
        
        # if the ad has already been shown and selected before 
        if (numbers_of_selections[i] > 0):
            # Compute the average reward for this ad up to this round - total rewards/total selections
            avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
            
            # Confidence interval for this ad [avg - delta, avg + delta]
            # n + 1 to account for zero-indexing in Python
            delta_i = np.sqrt(3/2 * np.log(n + 1) / numbers_of_selections[i])
            
            # Compute the **upper** confidence bound for this ad
            upper_bound = avg_reward + delta_i

        # if no prior selection, assume a very large upper bound. This 
        # is a very complicated way of ensuring that at each ad is selected at
        # least to give some historical data to our algo to base future selections on
        else:
            upper_bound = 1e400
            
        # Update the max upper bound and remember its corresponding ad
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ucb_ad_index = i
    
    # for this round, append the selected ad to the vector of selected ads
    ads_selected.append(ucb_ad_index)
    
    # update the number of times the ad has been selected
    numbers_of_selections[ucb_ad_index] += 1
    
    # compute the reward by comparing dataset's selected ad with ad shown by algo
    reward = dataset.values[n,ucb_ad_index] # if correct, this will be 1
    
    # update the sums of rewards for this ad
    sums_of_rewards[ucb_ad_index] += reward
    
    # update the final reward after all users have been shown the ad.
    total_reward += reward
    
"""
--------------------------------Interpretation-----------------------------------
- Doubled the total reward compared to random selection.
- For the first ten users, we showed the first 10 ads in sequence.
- This gave the algorithm information about how users responded to each of the
10 possible ads. The algo then used this data to refine its selection of ads
for future users.
- Towards the last few rounds, the algorithm consistently shows ad 5 to all users.
This means it has identified ad 4 as the one that will maximise reward/conversion rate.
(Index is 4 but ad is 5 because 0 indexing)
"""

"""
--------------------------------Visualization------------------------------------
"""
plt.figure(); plt.hist(ads_selected, edgecolor='black'); plt.xlabel("Advertisement");
plt.ylabel("Frequency of Selection"); plt.title("Ads Selection Count")

