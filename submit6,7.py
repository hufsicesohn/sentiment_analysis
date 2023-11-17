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


for i in range(32000):
  X_train[i] = re.sub('[^a-zA-Z]', ' ', X_train[i])

for i in range(32000):
  X_test[i] = re.sub('[^a-zA-Z]', ' ', X_test[i])


# In[ ]:


pipeline = Pipeline([
                     ('cnt_vect', CountVectorizer(stop_words = 'english', ngram_range=(1, 2))), 
                     ('lr_clf', LogisticRegression(C=10))])

# Pipeline 객체를 이용해 fit(), predict()로 학습/예측 수행, predict_proba()는 roc_auc 때문에 수행. 
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

submit6 = pd.read_csv('/sample_submission.csv')
submit6['sentiment'] = pred
submit6.head()


# In[ ]:


pipeline = Pipeline([
                     ('cnt_vect', TfidfVectorizer(stop_words = 'english', ngram_range=(1, 2))), 
                     ('lr_clf', LogisticRegression(C=10))])

# Pipeline 객체를 이용해 fit(), predict()로 학습/예측 수행, predict_proba()는 roc_auc 때문에 수행. 
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

submit7 = pd.read_csv('/sample_submission.csv')
submit7['sentiment'] = pred
submit7.head()


# In[ ]:


submit6.to_csv('/baseline_submit10.csv', index=False)
submit7.to_csv('/baseline_submit11.csv', index=False)
print('Done')


# In[ ]:


submit6 = pd.read_csv('/baseline_submit10.csv')
submit7 = pd.read_csv('/baseline_submit11.csv')

