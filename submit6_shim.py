#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')


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

train = pd.read_csv('/train.csv')
test = pd.read_csv('/test.csv')


train.text = train['text'].apply(data_processing)
train['text'] = train['text'].apply(lambda x: stemming(x))

test.text = test['text'].apply(data_processing)
test['text'] = test['text'].apply(lambda x: stemming(x))


# In[ ]:


X_train = train['text']
Y_train = train['sentiment']

X_test = test['text']


vect = CountVectorizer(ngram_range=(1,2)).fit(train['text'])
X_train = vect.transform(X_train)


svc = LinearSVC()
svc.fit(X_train, Y_train)


# In[ ]:


submit = pd.read_csv('./sample_submission.csv')

X_test = vect.transform(X_test)
preds = svc.predict(X_test)


submit['sentiment'] = preds
submit.head()
submit.to_csv('./baseline_submit10.csv', index=False)
print('Done')


# In[ ]:


submit6_심규상 = pd.read_csv('/baseline_submit10_심규상.csv')

