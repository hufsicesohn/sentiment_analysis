#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

train = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']


# In[ ]:


vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()

bnb = BernoulliNB()
bnb.fit(X_train, y_train)


# In[ ]:


X_test = vectorizer.transform(X_test).toarray()
preds = bnb.predict(X_test)

submit2 = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit2['sentiment'] = preds
submit2.head()


# In[ ]:


submit2.to_csv('C:/Users/sohnp/baseline_submit6.csv', index=False)
print('Done')


# In[ ]:


submit2 = pd.read_csv('C:/Users/sohnp/baseline_submit6.csv')

