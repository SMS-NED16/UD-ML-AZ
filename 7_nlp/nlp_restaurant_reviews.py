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


