""""
Data Preprocessing Template - a Python program that 
0. imports three necessary data analysis/viz libraries
1. reads data from a csv file
2. extracts the features and target variable into separate arrays
3. substitutes missing values in a column with the column's mean
4. encodes categorical data using one hot encoding
5. Standardizes/Normalizes the features (and target, if necessary) 
"""

"""IMPORTING REQUIRED LIBRARIES"""
import numpy as np 						# for optimised mathematical operations
import matplotlib.pyplot as plt   		# for graphs and visualizations
import pandas as pd  					# for storing and managing data


"""IMPORTING THE DATA"""
dataset = pd.read_csv('Data.csv') 		# must be in same directory as this program
print(dataset.head()) 					# check dataset loaded correctly


"""EXTRACTING FEATURES AND TARGET VARIABLE"""
X = dataset.iloc[:, :-1].values  		# numpy array, not a dataframe
y = dataset.iloc[:, -1].values  		# numpy array, not a dataframe


"""IMPUTATION - Substituting missing values in feature cols with statistical avg of col"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy="mean", axis=0)	# Create object
imputer = imputer.fit(X[:, 1:3])									# Fit object to specific cols
X[:, 1:3] = imputer.transform(X[:, 1:3])							# Perform transformation, store results


"""ENCODING CATEGORICAL DATA - LabelEncoder and OneHotEncoder"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# encode `Country` using LabelEncoder  - convert to numerical data
label_encoder_X = LabelEncoder()									# Instantiate object
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])					# Fitting and transforming in one step

# once converted to numerical values, eliminate relational order using OneHot
one_hot_encoder = OneHotEncoder(categorical_features=[0]) 			# Define the column index of the categorical feature
X = one_hot_encoder.fit_transform(X).toarray()						# transform, cast to array


"""ENCODNG TARGET VARIABLE - LabelEncoder. Relational order 
no longer relevant because this is a target variable, and not a feature"""
label_encoder_y = LabelEncoder()		# Instantiate new object for this 
y = label_encoder_y.fit_transform(y)	


"""TRAIN TEST SPLIT"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"""SCALING FEATURE - must be done for features with wide range or for some libraries"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()					# Instantiate separate objects for X and y
sc_X_train = StandardScaler.fit_transform(X=X_train)
sc_X_test = StandardScaler.transform(X=X_test)

# Don't need to scale target variable in this case, will need to do so for regression
# where the output variable takes a large range.