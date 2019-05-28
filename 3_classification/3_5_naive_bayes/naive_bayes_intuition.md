# Naive Bayes Classification 

## Bayes Theorem
- To understand Bayes theorem, we're going to use an analogy involving spanners.
- Assume we're at a factory where two machines produce spanners at different rates. The spanners are marked/tagged so we know which machine manufactured them.
- At the end of the day, workers pick out defective spanners from the pile.
- What's the probability of machine 2 producing a defective spanner?
- The mathematical concept that we'll be using to solve this problem is **Bayes Theorem**.
- Definition is P(A|B)  = P(B|A) * P(A) / P(B)
- We know that machine 1 produces 30 spanners/hr, machine 2 produces 20 spanners/hr. 
- Out of all the spanners produced, 1% are defective. 
- Out these defective spanners, we can see that 50% came from M1 and 50% from M2. 
- So what is the probability that a part produced by machine 2 is defective?

## Maths
- The probability that a random spanner comes from M1 is 30/50 = 0.6, and probability that the spanner came from M2 is 20/50 = 0.4.
- P(Defective) = 0.01, with a 50/50 split between M1 and M2.
- Given that a spanner is defective, the likelihood that the spanner came from M1 is 50%.
- Mathematically, this is written is P(M1 | Defective) = 0.5
- This reads as the probability that a spanner comes from M1 given that it is defective.
- We can write a similar statement for M2: P(M2 | Defective) = 50%
- So even though M2 accounts for 40% of the total output, it still accounts for 50% of all **defective** output.
- So P(Defective | M2 ) = P(M2 | Defective ) * P(Defective) / P(M2) = 0.5 * 0.01 / 0.4 = 0.005/0.04 = 0.0125.
- This means that if the factory produced 10,000 parts, 125 of them would have been defective ones produced by M2. 

## Intuition
- Suppose we had 1000 wrenches. We know that 400 came from M2 and 600 must have come from M1. 
- 1% have a defect - which means 1 of 10 spanners were defective.
- We know that 50% of the 10 defective spanners came from M2 i.e. total 5 defective wrenches from M2.
- So this means 5/400 = 0.0125 * 100% = 1.25% of defective spanners came from M2. 
- This is exactly what Bayes Theorem did in the previous section.

## Why Use The Bayes Theorem?
- If the items are labeled, why couldn't we just count the total defective spanners from M2 and divide them by the total spanners produced by M2?
- Firstly, this operation would be extremely time consuming, especially if the same operation has to be performed for every batch of 1000 wrenches.
- Secondly, in many cases we may not even have access to the actual information - the spanners may not be labeled. 
- Bayes theorem lets us calculate this with probabilities instead of actual raw data.

## Quiz - P(Defect | M1)
- P(Defect | M1) = P(M1 | Defective) * P(Defective) / P(M1) = 0.5 * 0.01 / 0.6 = 0.00833

## Naive Bayes Algorithm 
- Consider a 2D dataset with X1 = Age, X2 = Salary. 
- The categorical data we are trying to predict is whether a given person walks or drives to work.
- The Naive Bayes classifer helps us classify whether a new data point
- Steps	
	- Given the features of a new person, find the conditional probability that the person walks or drives.
	- Apply Bayes theorem to find P(Walks|X) = P(X|Walks) * P(X)/P(Walks)
		1. P(Walks) is the **prior probability**.
		2. P(X) is the **marginal likelihood**.
		3. P(X|Walks) is the **likelihood**.
		4. P(Walks|X) is the **posterior probability**.
	- Apply Bayes theorem to find P(Drives|X) = P(X|Drives) * P(X)/P(Drives) using a similar set of parameters.
	- Compare P(Walks | X) with P(Drives | X) and classify the new data point in the class with the greater posterior probability.

## Example
### Prior Probability
- Calculate the probability that a person walks: n_walks / (n_drives + n_walks) = number of walkers/number of observations = 10/30 = 1/3

### Marginal Likelihood
- To find **marginal likelihood**, create a region of specified radius on the dataset. Look at all the points inside this region are considered similar to the new data point X. So the radius of this region has a big effect on the way the algorithm works.
- P(X) is the probability of a new random data point being similar in features to the new point i.e. fall into the region.
- P(X) = Number of similar observations/total observations = 4 / 30

### Likelihood
- P(X|Walks)
- We're going to draw the same region from margin likelihood section: what is the probability that any randomly selected data point will be from this region **given** that the person walks to work?
- This means we'll only consider the data points in the circle that belong the 'walk' class.
- 4 data points in the region, 3 are walkers, so P(X|Walks) = Among those who walk/Total walkers = 3/10

### Posterior Probability
- Calculate using Bayes theorem: P(Walks | X) = 3/10 * 10/30 / 4/30 = 0.75
- 75% probability that a new random walker will be someone who has similar features as X.

### Repeat this for the P(Drives|X) (Is actually 0.25)
- P(Drives) = Drivers/Total Observations = 20/30 = 2/3
- P(X) = Number of similar observations in the region/Total observations = 4/30
- P(X|Drives) = Among those who drive/total drivers = 1/20
- P(Drives|X) = P(X|Drives) * P(Drives)/P(X) = 1/20 * 2/3 / 4/30 = **0.25**

### Compare Probabilities
- P(Walks|X) > P(Drives|X) = 0.75 > 0.25 => 75% chance that the person walks to work, so we will classify this data point as someone who walks to work.


## Additional Info on Naive Bayes Classifier
### Why "Naive"?
- It "naively" assumes that the features are independent. This assumption is often incorrect.
- Bayes theorem requires that the features/variables being used for prediction are independent.
- In our example, this is probably not the case: as people grow older, their salary usually increases. There is some correlation between the two variables, and they aren't necessarily independent.

### P(X)
- P(X) is the probability or likelihood that a randomly selected point from the dataset will exhibit features similar to the features in the region we've selected.
- P(X) = Number of Observations in the Region of Similarity/Total Observations
- This result will be the same for both classes. This means when comparing P(Walks|X) with P(Drives|X), we effectively compare P(X|Walks) * P(Walks) with P(X|Drives) * P(Drives) instead of evaluating the entire expression.
- This is only true if we're **comparing** the posterior probabilities, and not if we're calculating the actual values.

### More than 2 Classes?
- The example we considered was a binary classification problem.
- But when there are more than 2 classes in the dataset, the procedure remains essentially the same. 
- We calculate posterior probabilities for each class and then compare all probabilities, assigning the data point to the class with the maximum posterior probability.
- The sum of all posterior probabilities is 1, so with 2 classes we need only calculate 1 posterior probability and can derive the other one by subtracting from 1.
- With N classes, we will have to calculate at least N - 1 posterior probabilities.
