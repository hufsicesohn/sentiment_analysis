#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

train = pd.read_csv('/train.csv')
test = pd.read_csv('/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']


# In[ ]:


vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()

bnb = BernoulliNB()
bnb.fit(X_train, y_train)


# In[ ]:


X_test = vectorizer.transform(X_test).toarray()
preds = bnb.predict(X_test)

submit4 = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit4['sentiment'] = preds
submit4.head()


# In[ ]:


submit4.to_csv('C:/Users/sohnp/baseline_submit6.csv', index=False)
print('Done')


# In[ ]:


submit4 = pd.read_csv('C:/Users/sohnp/baseline_submit8.csv')

