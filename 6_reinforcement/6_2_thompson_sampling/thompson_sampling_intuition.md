# Thompson Sampling
- Another reinforcement learning algorithm which we will use to solve a variation of the multi-armed bandit problem in the context of an ad selection scenario.
- Goal is still to find balance between **exploration and exploitation**. 
- Similar, but not identical, to the upper confidence bound algorithm.
- Details of the formal definition of the multi-armed bandit problem remains the same.

## Bayesian Inference
- Ad `i` gets reward `y` from the Bernoulli Distribution `p(y | theta_i) ~ Bernoulli(theta_i)`
- `theta_i` is unknown, but we set its uncertainty by assuming it has a uniform distribution `p(theta_i) ~ Uniform([0, 1])` which is the **prior distribution**.
- Using Bayes rule, we approach `theta_i` by the posterior distribution
`p(theta_i | y) = p(y | theta_i) * p(theta_i)/ integral[ p(y | theta_i) dtheta]`
- This is proportional to the likelihood function `p(y|theta_i) \times p(theta_i)`
- We get `p(theta_i|y) ~ Bernoulli(number of successes + 1, number of failures + 1)`
- At each round `n` we take a random draw `theta_i(n)` from this posterior distribution `p(theta_i|y)` for each ad `i`.
- At each round `n` we select the ad `i` that has the highest `theta_i(n)`.

## Steps
1. At each round `n`, we consider two numbers for each ad `i`
	- `N1_i(n)` - the number of times the ad `i` got reward 1 up to round `n`.
	- `N0_i(n)` - the number of times the ad `i` got reward 0 up to round `n`.

2. For each ad `i`, we take a random draw fro mthe distribution below
	- `theta_i(n) = Bernoulli(N1_i(n) + 1, N0_i(n) + 1)`

3. We select the ad that has the highest `theta_i(n)`.


## Intuition
- Consider the distribution of three bandits. BGY each represent one bandit with its own distribution and have increasingly higher returns. 
	- The x axis shows the average value of the return for each bandit.
	- The y axis shows the number of returns for each bandit.
- Initially, we don't know anything about any of the bandits. 
	- We do some trial runs for each bandit and find that the returns are scattered around its mean value on the x-axis.
	- This creates a distribution for that machine.
	- This process is repeated for each bandit.
- We're constructing distributions of where we think the actual expected value might lie. 
	- The means of the distributions are where we **think** the mean values can be. It is the **predicted** Expected Mean Value.
	- The vertical bars on this graph show the **actual** expected mean value. 
	- However, the actual expected value can lie **anywhere** in the region of the distribution. 
- Using the trial runs, we've created a virtual/imaginary distributions and expected values for each bandit. 
- We then pull the lever for the machine with the highest **predicted** expected value. The new value will be closer to the **actual** expected value.
- Adding an observation to a machines results
	- narrows the range of its distribution
	- brings the predicted expected value closer to the actual expected value.
	- skews the distributions so that they are closer to the actual distribution.
- Repeat this process of testing all bandit, recording results, creating our own configuration of bandit distributions, identifying highest expected return, pull its lever, optimise the distribution. 