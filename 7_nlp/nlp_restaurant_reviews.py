#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 07:11:53 2019

@author: saadmashkoor
"""

# import libraries
import pandas as pd

# import the dataset - tab separated value (TSV) file because review text may have
# commas, pd.read_csv() would interpret this as delimiter, unlikely with tab
# quoting param ignores double quotes at tend of separator
dataset = pd.read_csv('Restaurant_Reviews.tsv', sep="\t", quoting=3)


"""-------------------------Cleaning Text - One Review-------------------------"""
"""To maximise efficiency of our NLP algorithm and create a small, relevant bag of
words, we will
1. Remove all punctuation from reviews.
2. Remove all stopwords such as 'the', 'a', 'it, that don't add context to the review.
3. Remove all numbers.
4. Perform Stemming - extract 'love' from 'loved', 'loving', 'lovely' - smaller bag, same info
"""

# Test cleaning procedure on first review
import re                               # regex library
review = dataset['Review'][0]           # extract first review for testing
review = re.sub('[^a-zA-Z]',            # regex that says don't remove lowercase/uppercase letters
                ' ',                    # removed characters will be replaced by space
                review)                 # string that we want to clean

# Convert all letters in review to lowercase
review = review.lower()

# Remove all stopwords - 'a', 'it', 'this' - irrelevant to our ML model, sparse matrix
import nltk                             # Natural Language Toolkit Library
#nltk.download('stopwords')             # Download list of stopwords built into NLTK
from nltk.corpus import stopwords       # Import the stopwords for use in this program

# Split the review into a list of individual words
review = review.split()

# List comprehension to remove any words in review also present in NLTK stopwords
# `set` ensures that comparison is with unique stopwords only - improves algo speed.
review = [word for word in review if not word in set(stopwords.words('english'))]

# Stemming - store only the root of the word - `lovely`, `loved`, `loving` all stem from `love`
# All versions of same word - not a sparse word mat - algo will be slow
from nltk.stem.porter import PorterStemmer              # import required class
port_stemmer = PorterStemmer()                          # instantiate Stemmer obj
review = [port_stemmer.stem(word) for word in review]   # could also do this in prev list comprehension

# `loved` in the previous list will now become `love` - root word

# Recombine words in review list into a string - words separated by space ' '
review = ' '.join(review)               # 'wow love place'

"""-------------------------Cleaning Text - All Reviews-------------------------"""
# Using a for loop to clean and store all reviews in the dataset
corpus = []                             # NLP term for a collection of text
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in
              set(stopwords.words('english'))]
    review = ' '.join(review)
    
    # append cleaned review to corpus
    corpus.append(review)
    
"""-----------------------------Bag of Words Model----------------------------"""
"""
Create a new table with 1000 rows (one review per row) and columns are each unique
word in corpus. Each cell represents whether a specific word in the corpus
appeared in a specific review. As every review won't necessarily contain all words,
this will likely be a spares table - most cells will be 0. **Sparse Matrix**

The model helps us predict whether a review is positive/negative based on the 
frequency of each word in our bag of words. Each column corresponding to a specific
word represents an independent variable. Helps us create a classification model.
"""
# tokenizing the reviews - store counts of each unique word in all reviews
from sklearn.feature_extraction.text import CountVectorizer

# only store the most frequently occurring words in matrix of features 
# reduces sparsity, improves model accuracy by selecting most relevant words.
word_limit = 1500
count_vectorizer = CountVectorizer(max_features=word_limit)

# The matrix of tokenized word counts = classification model features
X = count_vectorizer.fit_transform(corpus).toarray()

# Store labels
y = dataset['Liked'].values


"""---------------------------Classification Model----------------------------"""
# As a rule of thumb, Naive Bayes is a good classifier for NLP problems
# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training Set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions using the test set
y_pred = nb_classifier.predict(X_test)

# Quantizing performance
from sklearn.metrics import confusion_matrix, classification_report
conf_mat_nb = confusion_matrix(y_test, y_pred)
class_rep_nb = classification_report(y_test, y_pred)


"""-------------------------------Interpretation---------------------------------"""
"""
Out of 200 reviews, the model made 146 correct predictions and 54 incorrect
predictions. Hadelin says this may be because we had only 800 reviews to train the
model, which means the overall result is not that bad."""