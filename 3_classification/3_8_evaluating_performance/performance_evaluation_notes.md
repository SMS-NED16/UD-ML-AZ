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

# Cumulate Accuracy Profile (CAP)
- Assume you're a data scientist in a store that sells clothes.
- From experience, you know that at most 10% of the total 100k customers will respond to an offer.
- Assume further that because we're using a random sample of customers, the number of purchases increases linearly with the total customers contacted.
- Gradient of 0.1.
- Can we somehow improve this trend i.e. get more customers to respond to our offer?
- Instead of sending the offer to random customers, we'll use a ML model to identify which customers are likely to purchase our product in response to the offer.
- This will be a simple classification problem which can make use of customer data. Could then use this model to select customers to mail the offer to. This will maximise purchases.
- So our response rate will be higher than with the randomly selected customers - theoretically, it is possible for the customer purchases to saturate.
- The closer we get to 100k, the closer our purchases will be to 10k.
- The gradually increasing line from 0 to 100k will be the Cumulative Additive Profile.
- If the model is random, it will follow the random trend. If it is a great model, it will follow the CAP.
- The CAP axes are usually defined in terms of percentages.
- Practical models may suffer from issues like multicollinearity or other issues and will be between the CAP curve and the random curve.
- The CAP is **not** the ideal model, it just represents a practically good model.
- The **ideal** model will be such that only 10% of customers selected will make 100% purchases.
- The customer selection model is so good that it identifies customers who will make the maximum purchases.
- It doesn't matter how many other customers we contact after this 10%: they cannot increase purchases because everything will already be sold out.

**A bad model will be closer to random, a poor model will be between random and CAP, a good model will follow the CAP curve or be closer to the ideal curve.**

## CAP is not the same as ROC
ROC (Receiver Operating Characteristic) curve has to do with specificity and selectivity (metrics that deal with Type 1 and Type 2 errors). 