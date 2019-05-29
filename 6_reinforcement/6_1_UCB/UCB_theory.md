# Upper Confidence Bound Algorithm


## Recap - Multi-Armed Bandit Problem
- A reinforcement learning algorithm that will be used to solve the multi-armed bandit problem.
- We have multiple slot machines. Can bet money on any one of them. Have to bet to maximise returns. 
- Need to combine exploration of optimal machine distribution along with exploitation to maximise money earned.
- Modern application is identifying the best ad from a set of possible alternatives.

## Formalizing the Problem
- We have `d` arms. For example, arms are ads that we display to users each time they connect to a web page.
- Each time a user connects to this web page, that makes a `round`.
- At each `round` `n`, we choose one ad to to display to the user.
- At each round `n`, ad `i` gives the reward `r_i(n)` which is between [0, 1].
- `r_i(n)` is 1 if the user clicked on the ad `i`, and 0 if the user did not.
- Our goal is to maximize the total reward we get over many rounds.

## UCB Algorithm
1. At each round `n`, we consider two numbers for each ad `i`.	
	- `N_i(n)` - the number of times the ad `i` was selected up to round `n`. 
	- `R_i(n)` - the sum of rewards of the ad `i` up to round `n`.
2. From these two numbers we compute
	- the average reward of ad `i` up to round `n` 
		r_i(n)' = R_i(n)/N_i(n) 
	- the confidence interval
		[r_i(n)' - delta_i(n), r_i(n)' + delta_i(n)'] at round n where
		delta_i(n) = sqrt(1.5 * log(n)/N_i(n))
3. We select the ad `i` that has the maximium UCB `r_i(n)' + delta_i(n)`.
 
## Intuition (this isn't intuitive at all lolololol)
- For argument's sake, assume we already know the distribution of each machine.
- We know D5 has the best distriubtion because it has the highest probabilities for successful rolls. 
- IRL we don't know this, and want to use UCB to help us identify this. 
- Assume a starting mean expected return/value for each distribution.
- The formulae defined earlier create a confidence band. The confidence band includes the actual expected value. The confidence band is very large initially because this maximises the chance that the actual expected value lies in the band.
- We then create a random expected value for each machine's distribution.
- We then pick the machine with the largest confidence bound. At first, all machines will have equally large confidence bounds, so we can use a random machine. 
- After using the machine, we were unsuccessful. We now have another observation that is added to the sample of observations for this machine which will drive down the observed average of this machine.
- The confidence interval also becomes smaller: we have an additional data point so we are more confident in the results of this machine.
- We then repeat this process. Choose a random a machine, run it, check the result. The result either increases or decreases the observed average and decreases the confidence bound.
- **TLDR**
	- Find the machine with the highest confidence bound.
	- Run this machine.
	- The result will increase or decrease the randomly assumed average value, bringing it closer to the actual expected value.
	- The additional result will also shrink the confidence interval.
	- Repeat the procedure until the confidence intervals have shrunk sufficiently. When this occurs, we will be confident that the expected and observed average values are sufficiently close to each other.
	- Then examine all distributions for the one with the highest (or lowest, depending on context) average/expected value.