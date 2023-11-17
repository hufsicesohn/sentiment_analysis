#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import re

train = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']


# In[ ]:


def preprocess(x):
  for i in range(32000):
    x[i] = re.sub(r'https\S+|www|\S+https\S+', '', x[i], flags=re.MULTILINE)
    x[i] = re.sub(r'\@w+|\#', '', x[i])
    x[i] = re.sub(r'[^\w\s]', '', x[i])

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


X_train = preprocess(X_train)
X_test = preprocess(X_test)


# In[ ]:


pipeline = Pipeline([
                     ('cnt_vect', TfidfVectorizer(stop_words = 'english', ngram_range=(1, 2))), 
                     ('lr_clf', LogisticRegression(C=10))])

# Pipeline 객체를 이용해 fit(), predict()로 학습/예측 수행, predict_proba()는 roc_auc 때문에 수행. 
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

submit8 = pd.read_csv('/sample_submission.csv')
submit8['sentiment'] = pred
submit8.head()


# In[ ]:


submit8.to_csv('/baseline_submit12.csv', index=False)
print('Done')


# In[ ]:


submit8 = pd.read_csv('/baseline_submit12.csv')

