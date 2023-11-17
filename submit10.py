#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# In[17]:


nltk.download('punkt')


# In[19]:


nltk.download('stopwords')


# In[55]:


nltk.download('wordnet')


# In[57]:


nltk.download('omw-1.4')


# In[98]:


train = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/test.csv')


# In[73]:


Train = train['text']
Test = test['text']


# In[101]:


def data_processing(text):
    text = re.sub(r"https\S+|www|\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    text = text.lower().split()
    stops = set(stopwords.words('english'))
    text = [word for word in text if not word in stops]
    
    lemmatizer = nltk.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    
    return " ".join(text)


# In[102]:


X_train = train['text'].apply(data_processing)
X_test =  test['text'].apply(data_processing)


# In[103]:


Y_train = train['sentiment']


# In[104]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

pipeline = Pipeline([
                     ('cnt_vect', CountVectorizer(stop_words = 'english', ngram_range=(1, 2))), 
                     ('lr_clf', LogisticRegression(multi_class='multinomial', solver='lbfgs',C=10))])

# Pipeline 객체를 이용해 fit(), predict()로 학습/예측 수행, predict_proba()는 roc_auc 때문에 수행. 
pipeline.fit(X_train, Y_train)
pred = pipeline.predict(X_test)


# In[105]:


submit10 = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit10['sentiment'] = pred
submit10.head()


# In[107]:


submit10.to_csv('C:/Users/sohnp/baseline_submit14.csv', index=False)
print('Done')


# In[108]:


submit10 = pd.read_csv('C:/Users/sohnp/baseline_submit14.csv')


# In[83]:


from sklearn.metrics import confusion_matrix, accuracy_score

test_cm=confusion_matrix(y_test,pred)
test_acc=accuracy_score(y_test, pred)

print(test_cm)
print('\n')
print('정확도\t{}%'.format(round(test_acc*100,2)))


# In[116]:


X_train[34]


# In[ ]:




