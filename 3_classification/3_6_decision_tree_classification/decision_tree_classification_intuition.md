# Decision Tree Classification

## Introduction
- In the regression section, we used decision trees to carry out regression: predicting a continuous value for a variable using existing data.
- That involved 
	- splitting the domain of features into successively smaller sections called leaves 
	- identifying the terminal leaf that a new data point belonged to
	- assigning the average or mean of all data points within the terminal leaf as the value of the new data point.
- Now we will use decision trees for **classification**

## CART
- Classification and Regression Trees
	- Classification Trees
		- help with classification 
		- work with categorical data
	- Regression Tress
		- help with regression
		- predict continuous values

## Classification Trees
- Consider two dimensional categorical data that has belongs to two classes: R and G.
- We make several splits
	- Split 1: X2 = 60
	- Split 2: X1 = 50
	- Split 3: X1 = 70
	- Split 4: X2 = 20
- The split is done to maxmimise the number of data points belonging to one class in the resulting regions.
- Actually, more complex mathematics such as cross entropy are involved.
- But for our purposes, it is sufficient to know that the split aims to maximise the number of data points belonging to a specific class in the terminal leaves.
- The decision tree is as follows
	- Split 1: X2 > 60?
		- YES: 
			- Split 2: X2 > 60, X1 < 50?
				- YES: Green (Terminal)
				- NO: Red (Terminal)
		- NO
			- Split 3: X2 < 60, X1 < 70?
				- YES: RED (Terminal)
				- NO: 
					- Split 4: X2 < 60, X1 > 70, X2 < 20?
						- YES: Red
						- NO: Green
- Terminal leaves will predict what colour or class a data point belongs to.
- If the decision tree has many levels, it isn't necessary to check each value individually. A lot of the classificaiton is done probabalistically. 
- Instead of proceeding to the terminal leaves, we can use probabilities to predict the class.
- Doesn't have to be limited to two features. Can also work for multidimensional dataset.

## Decision Tree History
- Have been around for a very long time.
- Were being replaced by more sophisticated methods. 
- Reborn with upgrades
	- random forest
	- gradient boosting
- A simple tool that isn't super powerful on its own, but when combined with other algorithms can be very useful e.g. kinect, iOS facial recognition both use random forests, which are based on decision trees.