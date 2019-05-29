# Apriori Intuition

An **association rule learning** algorithm. Helps us solve problems like "Because you purchased this product, you may also like..."

## Thought Experiment
- What is the similarity between pampers and beer?
- The story goes that a convenience store did some analytics around what products people tend to purchase (market basket studies). 
- Analysed thousands of transactions/customers buying habits.
- Found that around 6 PM in the late afternoon/evening, people who buy diapers also buy beer. 
- One plausible explanation
	- When the husband gets home, the wife asks him to go and get diapers for the baby.
	- Because it is after hours, he also buys beer while he's at the convenience store.
- This explanation was inferred from the data, and could help us decide how to group data in the store. 
- Since people buy these items together, it makes sense to group them together.
- But most stores do the opposite (just as they do with bread and milk): they keep items in separate areas, forcing the customer to walk through the entire store and (hopefully) pick up more items along the way.

## What is Apriori All About?
- People who did X also did W, Y, Z,...
- Looking for rules or associations between different kinds of data. 
- E.g. using a table of user IDs and the IDs of the movies they watch, we can infer that 
	- people who like M1 will also like M2 and M3
	- people who like M2 are also likely to like M4
- Some of these rules can be strong associations, others can be weak associations.
- Apriori helps us identify the strongest associations and build models with them. 
- The same rule can be applied to a market basket study of the food people ordered at restaurants.

## Apriori Anatomy
- Has three major parts
	1. **Support**
	2. **Confidence**
	3. **Lift**

## Apriori Support
- **Movie Recommendation**
	- Support (M) = # of user watchlists containing the movie M/# of user watchlists
	- Support for movie M is the number of users who watch movie M divided by the total number of users.
- **Market Basket Optimisation**
	- Support(I) = # of transactions containing I/total transactions
	- I is a specific item.
- If 10/100 people have seen a certain movie, the support is 10/100 = 10%

## Apriori Confidence
- **Movie Recommendation**
	- Confidence(M1 -> M2) = # of watchlists containing M1 AND M2/total number of watchlists containing M1
	- If 7 people have seen both M1 and M2 while 40 have seen M1 only, confidence = 7/40 = 17.5%
- **Market Basket Optimisation**
	- Confidence(I1 -> I2) = # of transactions containing I1 AND I1/total # of transactions containing I1

## Apriori Lift
- Similar to the Naive Bayes' classifier.
- The confidence divided by the support.
- **Lift = Confidence/Support**
- If we suggested M2 to a random person from a new population, what is the likelihood that they will like it? 
- We know this is 10%. 
- Can we prove this result using some prior knowledge (hence apriori)?
- If we suggested M2 to the a random person from a new population, what is the likelihood that they will like it **given that** they already like M1?
- Lift is 17.5%/10% = 1.75 = improvement in the likelihood of the users enjoying M2 if we recommended it on the basis of them liking M1.

## Apriori Algorithm Steps
1. Set up a minimal support and confidence. 
	- There are far too many combinations to parse, so the algorithm is slow.
	- We need to define a minimal support and confidence that it doesn't waste time or computational resources on combinations/products that has a low success rate.
	- This limits the resources used by the model.
	- Don't have to create a completely new model for something that has a success late of 20% on its own (support), or one that has a success rate of 22% with prior knowledge (confidence).
2. Take all the subsets in the transactions having higher support than minimum support.
3. Take all the rules of htese subsets having higher confidence than minimum confidence.
4. Sort the rules by decreasing lift. 

We the consider the top 3 or top 5 algorithms for implementing a model that can help inform business decisions.

Recommender systems such as those of Amazon/Netflix are a good use case of apriori algorithm, but they are much more complex. 

Apriori is much more straightforward. 