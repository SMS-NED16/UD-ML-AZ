"""upper_confidence_bound_rpt.py - Python implementation of upper confidence bound
reinforcement learning algorithm as a solution to a multi-armed bandit problem in the
context of selecting the best ad from a set of 10 alternatives.

Repeating the code for practice and reinforcing concepts.

The algorithm is as follows
1. First show each unique ad to a user at least once.
2. Record the number of times the users responded to each ad.
    - First check if the ad has been shown to the user before.
    - If it hasn't been shown to the user before, set max_upper_bound to 
    a very large number like 1e400.
    - Remember the index of this ad. 
    - This ensures that any other ads, even if they haven't been seen
    before, won't be selected as the ad to be shown, because their 
    upper bound will still be 1e400, and won't exceed 1e400, so the
    index of the ad that first had the max ucb (and thus index of
    ad selected) will remain the same.
    - This ensures all ads are shown the user at least once.
3. Once all ads have been shown to the user, compute a new UCB
using the formulae discussed in theory for all ads.
    - Remember the max ucb and the index of the ad that caused it.
4. Identify ad to show to user based on max ucb and ucb index.
5. Append the selected ad to ads_selected, increment number of selections
for this specific ad.
6. If the ad has been selected correctly, the user will also click on the
same ad. Access the ad selection data from the dataset for the nth test and
the ith ad. This is the reward (1 if true, 0 if false).
7. Increment reward for selected ad as well as total reward using 1/0
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Define parameters for UCB implementation
# Number of training rounds i.e. number of times an ad will be shown to a user
N_ROUNDS = 10000                # same as number of users in dataset
d = len(dataset.columns)        # number of columns/number of ads

# number of time each ad was selected to be show nto a user up to a given round
numbers_of_selections = [0] * d

# sum of rewards for each ad up to round n
sums_of_rewards = [0] * d

# List to store index of ad selected by algo in each round
ads_selected = []

# variable to store the total reward at the end of all iterations
total_reward = 0

# For every round of showing a user an ad
for n in range(0, N_ROUNDS):
    
    # create a variable to store max UCB - use this to identify best ad
    max_upper_bound = 0
    
    # variable to remember index of the ad with highest UCB
    ucb_ad_index = 0
    
    # for every ad in the list of possible ads
    for i in range(0, d):
        
        # if the ad has already been shown to the user before
        if (numbers_of_selections[i] > 0):
            
            # Compute the average reward for this ad so far
            avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
            
            # Confidence interval = avg +- delta
            delta_i = np.sqrt(3/2 * np.log(n + 1) / numbers_of_selections[i])
            
            # UCB = avg + delta
            ucb_i = avg_reward + delta_i
            
        # if the ad hasn't been shown before
        else:
            ucb_i = 1e400
            
        # Update the max upper bound and the index of the corresponding ad
        # this will later be used to identify the best ad to show to a user
        if (ucb_i > max_upper_bound):
            max_upper_bound = ucb_i
            ucb_ad_index = i  
            
    # for this round, append the selected ad to the `ads_selected` vector 
    # this vector keeps track of every ad that was selected by the algorithm 
    # to show to a user for each round
    ads_selected.append(ucb_ad_index)
    
    # Also update the selection count for this ad
    numbers_of_selections[ucb_ad_index] += 1
    
    # Compute the reward for this step by comparing user's selection with shown ads
    # if the algo chose the right ad, the ad selected by this user should be the 
    # same as the ad selected by the algo, and will be 1
    reward = dataset.values[n, ucb_ad_index]
    
    # update the total reward for this ad
    sums_of_rewards[ucb_ad_index] += reward
    
    # update the total reward
    total_reward += reward
    
    
# visualizing results
plt.figure(); plt.hist(ads_selected, edgecolor='black');
plt.xlabel('Ad Index'); plt.ylabel('Number of Times Selected');
plt.title('Number of Times Each Ad was Selected by UCB Algo');