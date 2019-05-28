# False Positives and Negatives

## Example - Logistic Regression Results
In our original logistic regression dataset, we made predictions by calculating the probability of a particular observation belonging to the positive class. If this probability was greater than 0.5, we projected it to class 1. Otherwise, we projected it to class 0.

However, there were many data points with actual values of 1 and predicted probabilities < 0.5, as well as those with actual values of 0 and predicted probabilities > 0.5 

In Kirill's diagram, we the projected prediction is the same as the actual prediction. 

However, observation 2 was predicted to be class 0 on the basis of probability, while the actual value of this observation was at 1. 

The same applise to observation 3: predicted was 1 but actual was 0.

## Types of Errors
- Type 1 Error: False Positive - Actual 0, Predicted 1
- Type 2 Error: False Negative - Actual 1, Predicted 0