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

# Confusion Matrix
A tool to help us summarise True Positives, True Negatives, False Positives, and False Negatives in the results of a classification model.

 		PREDICTED |		0	 | 		1	 |
		ACTUAL    | 
		  0				A  			B
		  1  			C 			D

A = Actual 0, Predicted 0 => True Negative
B = Actual 0, Predicted 1 => False Negative (Type 2)
C = Actual 1, Predicted 0 => False Positive (Type 1)
D = Actual 1, Predicted 1 => True Positive 

Some important metrics
- **Accuracy Rate AR** - Correct/Total = (A+D)/Total
- **Error Rate ER** - Incorrect/Totla = (B+C)/Total

# Accuracy Paradox
- Consider the following confusion matrix
		PREDICTED |		0	 | 		1	 |
		ACTUAL    | 
		  0		  |	  9.7k   |	   150   |
		  1  	  |		50 	 |     100   |

- Accuracy Rate = (9.7k + 100)/10k = 98%
- Assume that we modified our algorithm so that it only predicts 0 for all examples. 
		PREDICTED |		0	 | 		1	 |
		ACTUAL    | 
		  0		  |	    9850 |	   0     |
		  1  	  |		150  |     0     |
- Now the accuracy is 9850/10k = 98.5
- The accuracy rate increased by 0.5%, even though we basically stopped using the model.
- We're not really applying any kind of model to the dataset and are still getting an increased accuracy.
- This is the **Accuracy Paradox**.