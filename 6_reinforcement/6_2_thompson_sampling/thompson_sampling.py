#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 06:08:08 2019

@author: saadmashkoor
"""

"""thomspon_sampling.py - Solving a variation of the multi-armed bandit problem
in the context of ad selection using the Thomspon Sampling reinforcement learning
algorithm"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import random

# import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# defining parameters for Thompson Sampling - same as those for UCB
N = len(dataset)                            # number of rounds                       
d = len(dataset.columns)                    # number of ads shown in each round
ads_selected = []                           # ad selected in each round
total_reward = 0                            # total reward at end of all rounds

# new parameters - specific to Thompson Sampling
numbers_of_rewards_1 = [0] * d              # number of times each ad got reward 1 upto a round
numbers_of_rewards_0 = [0] * d              # number of times each ad got reward 0 upto a round

# for every user that will log into the social media site
for n in range(0, N):
    # index of ad selected for this round
    ad_index = 0
    
    # maximum expected value for this round
    max_random = 0
    
    
    # for every ad that can be shown to the user
    for i in range(0, d):    
        # compute the random posterior probability for this ad
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                         numbers_of_rewards_0[i] + 1)
        
        # update the maximum random posterior probability and corresponding ad index
        if random_beta > max_random:
            max_random = random_beta
            ad_index = i
        
    # append the index of the selected ad
    ads_selected.append(ad_index)
    
    # If the algo selected the right ad, this will index `1`, otherwise `0`
    reward = dataset.values[n, ad_index]
    
    # update total reward at end of this round
    total_reward += reward
    
    # increment the 0/1 reward count for this ad
    if reward == 1:
        numbers_of_rewards_1[ad_index] += 1
    else:
        numbers_of_rewards_0[ad_index] += 1

# visualizing results
plt.figure(); plt.hist(ads_selected, edgecolor='black');
plt.xlabel("Ad Index"); plt.ylabel("Number of Times Selected");
plt.title("Ad Selection Count - Thompson Sampling")