# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:36:13 2019

@author: aliag
"""

import re
import pandas as pd
import RAKE

dt = pd.read_csv('Train.csv', nrows=5000)

dt['Text'] = dt['Title'] + dt['Body']

# function definition for cleansing the text


def cleanse_text(text):

    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>", " <> ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    return text


dt['Text'] = dt['Title'] + dt['Body']
dt['Text'] = dt['Text'].apply(lambda x: cleanse_text(x))

stop_dir = "stopwords.txt"
rake_object = RAKE.Rake(stop_dir)

# Adding new column to the dataframe
dt['Keywords'] = ""

# getting the keywords out
for i in range(len(dt)):
    keywords = rake_object.run(dt['Text'][i])
    dt.at[i, 'Keywords'] = keywords[0]
