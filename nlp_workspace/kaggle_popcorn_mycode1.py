'''
https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words
'''

#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import re
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from nlputils import *

if __name__ == '__main__':
    # TODO 너무 오래 걸린다 일단 일부만 가지고 하자 전체 할때는 이거 지우기
    train = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3, nrows=100)

    # train = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('./data/testData.tsv', header=0, delimiter="\t", quoting=3 )

    print('The first review is:')
    print(train['review'][0])

    stop_words = stopwords.words('english')

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    print("Cleaning and parsing the training set movie reviews...\n")
    for review in train['review']:
        tokens = get_token_en(review)
        clean_review = get_clean_words_en(tokens, stop_words)
        clean_train_reviews.append(clean_review)



    # ****** Create a bag of words from the training set
    #
    print("Creating the bag of words...\n")
    # BoW_list = [tokens_to_bow(tokens, language='en') for tokens in clean_train_reviews]
    # print(BoW_list[0])
    cv = CountVectorizer()
    cv.fit(' '.join(clean_train_reviews[0]))

    print(cv.transform([clean_train_reviews[0]]).toarray())

    # ******* Train a random forest using the bag of words
    #
    print("Training the random forest (this may take a while)...")



    print("Cleaning and parsing the test set movie reviews...\n")

    # Use the random forest to make sentiment label predictions
    print("Predicting test labels...\n")


    print("Wrote results to Bag_of_Words_model.csv")