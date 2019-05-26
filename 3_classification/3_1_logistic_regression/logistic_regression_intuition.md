# Logistic Regression Intuition

## Regression vs Classification
- We already know about regression: single and multiple linear regression.
- We can use regression techniques to predict a continuous quantity based on a number of independent features.
- These techniques are useful for problem such as predicting lifespan, forecasting housing prices, or estimating similar continuously varying quantities.
- But what about the following problems?
	- Assessing whether a given customer will respond positively to a promotional email?
- The independent variable is `Age` while the output is a whether or not the customer responded and took `Action` (Y/N).
- The scatterplot for this dataset is somewhat similar to earlier problems, but the data is too separated for a linear regression to model this properly.
- Instead of predicting **exactly** whether a customer will respond to the material, we can try to predict the **probability** that the customer will respond to the promotional material. 
- If the probability is above a certain threshold, we can assume they will respond to the mailing (class = 1) and otherwise we can assume that they won't (class = 0).
- A linear regression model is not the best choice for this
	- It follows a straight line trend between the two class clusters.
	- It can exceed 1 and drop below 0: this doesn't make sense as this means there are some ages for which the user is more than 100% likely to respond to the mailing, and others for which the user has a negative probability of responding to the mailing.
- So the linear regression model needs to be modified so that 
	- it is bounded by $[0, 1]$
	- It varies non-linearly between the $[0, 1]$.

## Sigmoid Functions and Logistic Model
- Our original linear regression model is $y = b_0 + b_1x$
- If we consider the sigmoid function $$p = \frac{1}{1 + e^{-y}}$$
- Substitute $y$ in the sigmoid function and solve for $y$ to get $$ ln(\frac{p}{1-p} = b_0 + b_1x)$$
- This is the formula for **logistic regression**. Converts the trend from linear to **S-shaped**.
- This is the line of best fit that fits our observations and can assist with classification. 

## Logistic Regression for Classification
- We can make classifications based on the **predicted probability** that the output will be 1 or 0 for a given set of features.
- If the predicted probability for a given set of features is above 50%/0.5, project the probability
to absolute certainty (positive class).
- If the predicted probability for a given set of features is below 50%, project the probability to
0 (negative class).