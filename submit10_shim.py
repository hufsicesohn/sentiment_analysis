#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag


# In[2]:


nltk.download('punkt')


# In[3]:


nltk.download('stopwords')


# In[43]:


nltk.download('averaged_perceptron_tagger')


# In[107]:


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


# In[108]:


train = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/test.csv')

X_train = train['text'].apply(data_processing)
X_train = X_train.apply(lambda x: stemming(x))

X_test =  test['text'].apply(data_processing)
X_test = X_test.apply(lambda x: stemming(x))

Y_train = train['sentiment']


# In[109]:


def tokenize_text(text):
    tokens = word_tokenize(text)
    
    return tokens

text_series = X_train
token_list = []

for text in text_series:
    tokens = tokenize_text(text)
    token_list.append(tokens)
    
tokens = pd.Series(token_list)


# In[110]:


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


# In[111]:


clean_token_list = []

for token in tokens:
    clean_token = list(filter(lambda x: x not in common_words, token))
    clean_token_list.append(clean_token)

clean_tokens = pd.Series(clean_token_list)


# In[112]:


## 토큰에 tag붙임. 이거 작업 좀 걸림.

tagged_tokens = clean_tokens.apply(lambda tokens: pos_tag(tokens))


## CD(숫자를 나타내는 품사), NNP(고유명사. 단수형), NNPS(고유명사, 복수형) tag 제거 후 Series객체로 반환

filtered_series = [[word for word, tag in tagged_tokens if tag not in ['CD','NNP','NNPS']] for tagged_tokens in tagged_tokens]
filtered_series = pd.Series(filtered_series)


## 토큰화 되어있는 Series객체 토큰들 합치기

X_train = filtered_series.apply(lambda tokens:' '.join(tokens))


# In[102]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.5,random_state=130)


# In[103]:


x_train


# In[105]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,2)).fit(x_train)
x_train_vect = vect.transform(x_train)

clf = RandomForestClassifier(n_estimators=500,min_samples_split=10)
clf.fit(x_train_vect, y_train)


# In[106]:


from sklearn.metrics import confusion_matrix, accuracy_score

x_test_vect = vect.transform(x_test)
preds = clf.predict(x_test_vect)

test_cm=confusion_matrix(y_test,preds)
test_acc=accuracy_score(y_test, preds)

print(test_cm)
print('\n')
print('정확도\t{}%'.format(round(test_acc*100,2)))


# In[63]:


X_test = vect.transform(X_test)
preds = log.predict(X_test)
submit10_shim = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit10_shim['sentiment'] = preds
submit10_shim.head()


# In[64]:


submit10_shim.to_csv('C:/Users/sohnp/baseline_submit14_심규상.csv', index=False)
print('Done')


# In[ ]:




