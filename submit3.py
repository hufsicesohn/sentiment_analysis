#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']


# In[ ]:


vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()

clf = RandomForestClassifier(n_estimators=100,min_samples_split=5)
clf.fit(X_train,y_train)


# In[ ]:


X_test = vectorizer.transform(X_test).toarray()
pred = clf.predict(X_test)

submit3 = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit3['sentiment'] = preds
submit3.head()


# In[ ]:


submit3.to_csv('C:/Users/sohnp/baseline_submit7.csv', index=False)
print('Done')


# In[ ]:


submit3 = pd.read_csv('C:/Users/sohnp/baseline_submit7.csv')

