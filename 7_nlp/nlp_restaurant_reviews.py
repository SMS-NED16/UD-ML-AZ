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


"""---------------------------Cleaning the Texts------------------------------"""
"""To maximise efficiency of our NLP algorithm and create a small, relevant bag of
words, we will
1. Remove all punctuation from reviews.
2. Remove all words such as 'the', 'a', 'it, that don't add context to the review.
3. Remove all numbers.
4. Perform Stemming - extract 'love' from 'loved', 'loving', 'lovely' - smaller bag
"""

# Test cleaning procedure on first review
import re                               # regex library
review = dataset['Review'][0]           # extract first review for testing
review = re.sub('[^a-zA-Z]',            # regex that says don't remove lowercase/uppercase letters
                ' ',                    # removed characters will be replaced by space
                review)                 # string that we want to clean

