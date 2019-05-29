# Upper Confidence Bound - Theory

## Overview
- Reinforcement learning (RL) is an advanced class of ML algos that is used to train systems by providing them a set of actions and rewarding actions which bring the system closer to a given goal.
- RL is used to teach robot dogs to walk.
	- Instead of explicitly programming the steps the robot dog needs to take to walk, we could implement a RL algo.
	- This means providing the dog with a set of actions that it can take and set it a goal: to take a step forwward without falling.
	- Successfully executing this goal earns the dog a reward, failing will earn it a penalty.
	- The dog will try all the possible actions and eventually converge on the proper set of actions it needs to take to walk.
- This is a very advanced topic and could be an entire course on its own.
- As an introduction to this class of algos, we will attempt to solve the multi-armed bandit problem.
- But this is by no means the only problem that can be solved with reinforcement learning.

## The Multi-Armed Bandit Problem
- A one-armed bandit is a slot machine with a single lever that can be pulled to randomly roll slots.
- A specific combination of slots will earn a prize. 
- One of the quickest way to lose money in a casino since 50% chance of not winning anything, so any money spent on operating the machine is lost.
- The **multi-armed bandit** problem is the challenge a person faces when he/she has to deal with multiple one-armed bandit machines (usually 5).
- How does the user operate them to maximise the return from the machines?
- The assumption is that each machine has a distribution of numbers/outcomes out of which the machine picks results. The distribution is usually different for each machine. 
- The problem is that we don't know in advance what these distributions are.
- Our **goal** is to figure our which of these distributions is best suited for maximising training. 
- The most left-skewed distribution has the most favourable outcomes (mean/median/mode) so if we knew which machine had this distribution, we'd spend all our money on that machine to maximise earning.
- The left-skewed distribution means the probability of getting a successful combination is higher.
- The longer we spend to figure out the distributions, the more money we are likely to waste.
- **Tradeoff: Expectation vs Exploration**: the longer we spend on exploring the distributions of the machines, the more money we are likely to spend/lose. However, we're also more likely to find the best distribution that could maximise our earning. 


## Case Study - Coca Cola Ad Campaign
- Selecting the best ad for highest return on investment is an example of the multi-armed bandit problem.
- One way to solve this problem is to run multiple A/B tests and wait until we have a large enough sample to conclude which ad is the best.
	- However, this will take a lot of time and money.
	- This is pure explanation.
- The challenge is to exploit the best ad **in the process** of the actual exploration. 

## UCB Intuition