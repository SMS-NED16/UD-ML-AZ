# Comparing UCB and Thompson Algorithm

## UCB 
- **Determinstic**: Does not rely on probabilities or expected values. Just need to check the bandit with the highest upper confidence bound. The steps that this algorithm takes are not affected by randomness. All I/Os are exactly known and can be exactly calculated.
- **Update at Every Round**: When we get a value from the machine after an exploration, we must incorporate it into the upper bound based value calculation. We can't proceed to the next step without updating the upper confidence bound. Unless the UCB is updated at each step, it would give the exact same result every time. So the results of the previous round are required for the exploration/exploitation in the next one.
-

## Thompson
- **Probabilistic**: Based on expected and actual distributions for each bandit. Each round of the Thompson Sampling algorithm draws from the Bernoulli distribution/our own distribution, which will result in different values every time. 
- **Delayed Feedback**: Can accommodate delayed feedback. Don't need to update distributions after every sampling operation. We generate a new set of hypothetical bandit distributions at each step, and don't necessarily need values from previously generated distributions to create new ones. This makes it less computationally intensive but can affect accuracy. 