#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.model_selection import train_test_split


# In[21]:


nltk.download('all')


# In[22]:


stop_words = set(stopwords.words('english'))

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www|\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    
    return " ".join(filtered_text)


stemmer = PorterStemmer()

def stemming(data):
    text = [stemmer.stem(word) for word in data]
    
    return data


# In[23]:


train = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/test.csv')

X_train = train['text'].apply(data_processing)
X_train = X_train.apply(lambda x: stemming(x))

X_test =  test['text'].apply(data_processing)
X_test = X_test.apply(lambda x: stemming(x))

Y_train = train['sentiment']


# In[24]:


def tokenize_text(text):
    tokens = word_tokenize(text)
    
    return tokens

text_series = X_train
token_list = []

for text in text_series:
    tokens = tokenize_text(text)
    token_list.append(tokens)
    
tokens = pd.Series(token_list)


# In[25]:


def get_freq_count(tokens):
    neg_token = tokens[train['sentiment'] == 2]
    pos_token = tokens[train['sentiment'] == 1]
    neul_token = tokens[train['sentiment'] == 0]
    neg_freq = pd.Series(np.concatenate([w for w in neg_token])).value_counts()
    pos_freq = pd.Series(np.concatenate([w for w in pos_token])).value_counts()
    neul_freq = pd.Series(np.concatenate([w for w in neul_token])).value_counts()
    
    return neg_freq, pos_freq, neul_freq

def remove_doubled_words(neg_freq ,pos_freq, neul_freq, tokens):
    top_50_neg = neg_freq[:20]
    top_50_pos = pos_freq[:20]
    top_50_neul = neul_freq[:20]
    remove_word = [p for p in top_50_pos.index if p in top_50_neg.index]
    tokens_removed = remove_stop_words(tokens, remove_word)
    final_tokens = cleaning_tokens(tokens_removed)
    
    return final_tokens



## 셋이 공통적으로 겹치는 부분 제거, 원래 3개 지웠는데 im만 지운게 성능 더 좋아서 im 만 지움.

neg_freq ,pos_freq, neul_freq = get_freq_count(tokens)

top_50_pos = pos_freq[:3]
top_50_neg = neg_freq[:3]
top_50_neul = neul_freq[:3]

common_words = [p for p in top_50_neg.index if p in set.intersection(set(top_50_pos.index), set(top_50_neul.index))]


# In[27]:


clean_token_list = []

for token in tokens:
    clean_token = list(filter(lambda x: x not in common_words, token))
    clean_token_list.append(clean_token)

clean_tokens = pd.Series(clean_token_list)


# In[28]:


## 토큰에 tag붙임. 이거 작업 좀 걸림.

tagged_tokens = clean_tokens.apply(lambda tokens: pos_tag(tokens))


## CD(숫자를 나타내는 품사), NNP(고유명사. 단수형), NNPS(고유명사, 복수형) tag 제거 후 Series객체로 반환

filtered_series = [[word for word, tag in tagged_tokens if tag not in ['CD','NNP','NNPS','PRP','PRP$','DT','IN']] for tagged_tokens in tagged_tokens]
filtered_series = pd.Series(filtered_series)


## 토큰화 되어있는 Series객체 토큰들 합치기

X_train = filtered_series.apply(lambda tokens:' '.join(tokens))


# In[31]:


Y_train.value_counts()


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE


vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

smote = SMOTE(random_state=67)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_vec, Y_train)


# 랜덤 포레스트 분류기
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=89)

# 그래디언트 부스팅 분류기
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=4)


ensemble_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('gb', gb_classifier)
], voting='hard')


ensemble_classifier.fit(X_train_resampled, Y_train_resampled)



Y_test_pred = ensemble_classifier.predict(X_test_vec)

# 결과 저장
submit13 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/sample_submission.csv')
submit13['sentiment'] = Y_test_pred
submit13.to_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment7.csv', index=False)

print('Done')


# In[33]:


submit13 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment7.csv')


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 랜덤 포레스트 분류기
weights = {0:3.0, 1:2.0, 2:1.0}
rf_classifier = RandomForestClassifier(n_estimators=1000, class_weight=weights, random_state=47)

# 그래디언트 부스팅 분류기
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=47)


ensemble_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('gb', gb_classifier)
], voting='hard')


ensemble_classifier.fit(X_train_vec, Y_train)



Y_test_pred = ensemble_classifier.predict(X_test_vec)

# 결과 저장
submit14 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/sample_submission.csv')
submit14['sentiment'] = Y_test_pred
submit14.to_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment8.csv', index=False)

print('Done')


# In[34]:


submit14 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment8.csv')


# In[40]:


class_counts = Y_train.value_counts()
class_weights = 1./class_counts
class_weights = class_weights/class_weights.min()
class_weights = class_weights.to_dict()


# In[41]:


class_weights


# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train).astype('float32')
X_test_tfidf = tfidf.transform(X_test).astype('float32')


# In[47]:


def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=15184, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model


# In[49]:


import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential

from scipy.sparse import hstack, vstack

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm_notebook as tm
import re

df_len = X_train_tfidf.shape[0]
pred = []
for i in range(10):
    
    model = build_model()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(0.001),
                  metrics=['accuracy'])
    
    history = model.fit(x = vstack((X_train_tfidf[:int(df_len/10*(i))], X_train_tfidf[int(df_len/10*(i+1)):])),
                        y = np.concatenate([Y_train[:int(df_len/10*(i))], Y_train[int(df_len/10*(i+1)):]], axis = 0),
                        validation_data = (X_train_tfidf[int(df_len/10*i):int(df_len/10*(i+1))],
                                          Y_train[int(df_len/10*i):int(df_len/10*(i+1))]),
                        epochs = 4)
    
    pred_ = model.predict(X_test_tfidf)
    
    #pred.append(np.argmax(pred_, axis = 1))
    pred.append(pred_)


# In[ ]:




