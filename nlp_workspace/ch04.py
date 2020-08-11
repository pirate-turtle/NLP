import re
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def preprocessing(review):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]

    return (words)


train_data = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test_data = pd.read_csv('./data/testData.tsv', header=0, delimiter="\t", quoting=3 )

clean_train_reviews = []
for review in train_data['review']:
    clean_train_reviews.append(preprocessing(review))

print(clean_train_reviews[0])


clean_train_df = pd.DataFrame({'review': clean_train_reviews, 'sentiment': train_data['sentiment']})

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_reviews)
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)

print(text_sequences[0])


word_vocab = tokenizer.word_index
print(word_vocab)

print("전체 단어 개수: ", len(word_vocab))


# 사전 만들기
data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) + 1


MAX_SEQUENCE_LENGTH = 174

train_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print('Shape of train data: ', train_inputs.shape)