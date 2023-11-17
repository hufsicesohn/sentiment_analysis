#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB
import re
import nltk

train = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']


# In[ ]:


def preprocess(x):
  for i in range(32000):
    x[i] = re.sub('[^a-zA-Z]', ' ', x[i])

  for i in range(32000):
    x[i] = x[i].lower().split()

  for i in range(32000):
    stops = set(stopwords.words('english'))
    x[i] = [word for word in x[i] if not word in stops]

  for i in range(32000):
    stemmer = nltk.stem.SnowballStemmer('english')
    x[i] = [stemmer.stem(word) for word in x[i]]

  for i in range(32000):
    x[i] = ' '.join(x[i])

  return x


# In[ ]:


def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www|\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[ ]:


X_train = preprocess(X_train)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()

bnb = BernoulliNB()
bnb.fit(X_train, y_train)


# In[ ]:


X_test = preprocess(X_test)
X_test = vectorizer.transform(X_test).toarray()
preds = bnb.predict(X_test)

submit5 = pd.read_csv('/sample_submission.csv')
submit5['sentiment'] = preds
submit5.head()


# In[ ]:


submit5.to_csv('C:/Users/sohnp/baseline_submit9.csv', index=False)
print('Done')


# In[ ]:


submit5 = pd.read_csv('C:/Users/sohnp/baseline_submit9.csv')

