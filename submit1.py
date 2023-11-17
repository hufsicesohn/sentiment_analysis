#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']


# In[ ]:


vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()

classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:


X_test = vectorizer.transform(X_test).toarray()
preds = classifier.predict(X_test)
submit = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit['sentiment'] = preds
submit.head()


# In[ ]:


submit.to_csv('C:/Users/sohnp/baseline_submit5.csv', index=False)
print('Done')


# In[ ]:


submit = pd.read_csv('C:/Users/sohnp/baseline_submit5.csv')

