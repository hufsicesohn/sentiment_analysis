#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.model_selection import train_test_split


# In[2]:


nltk.download('punkt')


# In[64]:


get_ipython().system('python -m spacy download en')


# In[61]:


pip install spacy


# In[40]:


nltk.download('sentiwordnet')


# In[41]:


nltk.download('wordnet')


# In[3]:


nltk.download('stopwords')


# In[43]:


nltk.download('averaged_perceptron_tagger')


# In[2]:


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


# In[105]:


train = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/test.csv')

X_train = train['text'].apply(data_processing)
X_train = X_train.apply(lambda x: stemming(x))

X_test =  test['text'].apply(data_processing)
X_test = X_test.apply(lambda x: stemming(x))

Y_train = train['sentiment']


# In[106]:


def tokenize_text(text):
    tokens = word_tokenize(text)
    
    return tokens

text_series = X_train
token_list = []

for text in text_series:
    tokens = tokenize_text(text)
    token_list.append(tokens)
    
tokens = pd.Series(token_list)


# In[107]:


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


# In[47]:


import nltk
from nltk.corpus import sentiwordnet as swn

list(swn.senti_synsets('disappointment', 'n'))[0].neg_score()


# In[118]:


neg_freq.head(10)


# In[76]:


clean_token_list = []
common_neg_words = ['like', 'get', 'go']

for i in range(len(tokens)):
    token = tokens.iloc[i]  # 각 토큰 가져오기
    sentiment = train['sentiment'].iloc[i]  # 해당 토큰의 sentiment 값 가져오기
    
    # 해당 토큰의 sentiment가 2인 경우 common_neg_words를 사용하여 필터링
    if sentiment == 2:
        clean_token_neg = list(filter(lambda x: x not in common_neg_words, token))
        clean_token = list(filter(lambda x: x not in common_words, clean_token_neg))
    else:
        clean_token = list(filter(lambda x: x not in common_words, token))
    
    clean_token_list.append(clean_token)

clean_tokens = pd.Series(clean_token_list)


# In[108]:


## 토큰에 tag붙임. 이거 작업 좀 걸림.

tagged_tokens = clean_tokens.apply(lambda tokens: pos_tag(tokens))


## CD(숫자를 나타내는 품사), NNP(고유명사. 단수형), NNPS(고유명사, 복수형) tag 제거 후 Series객체로 반환

filtered_series = [[word for word, tag in tagged_tokens if tag not in ['CD','NNP','NNPS','PRP','PRP$','DT','IN']] for tagged_tokens in tagged_tokens]
filtered_series = pd.Series(filtered_series)


## 토큰화 되어있는 Series객체 토큰들 합치기

X_train = filtered_series.apply(lambda tokens:' '.join(tokens))


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.5,random_state=130)


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE


vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

smote = SMOTE(random_state=50)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_vec, Y_train)


# 랜덤 포레스트 분류기
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=4)

# 그래디언트 부스팅 분류기
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=4)


ensemble_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('gb', gb_classifier)
], voting='hard')


ensemble_classifier.fit(X_train_vec, Y_train)



Y_test_pred = ensemble_classifier.predict(X_test_vec)

# 결과 저장
submit11 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/sample_submission.csv')
submit11['sentiment'] = Y_test_pred
submit11.to_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment_test.csv', index=False)

print('Done')


# In[ ]:


submit11 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment_test.csv')


# In[109]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE


vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

smote = SMOTE(random_state=50)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_vec, Y_train)


# 랜덤 포레스트 분류기
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=4)

# 그래디언트 부스팅 분류기
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=4)


ensemble_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('gb', gb_classifier)
], voting='hard')


ensemble_classifier.fit(X_train_vec, Y_train)



Y_test_pred = ensemble_classifier.predict(X_test_vec)

# 결과 저장
submit12 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/sample_submission.csv')
submit12['sentiment'] = Y_test_pred
submit12.to_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment6.csv', index=False)

print('Done')


# In[110]:


submit12 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment6.csv')


# In[ ]:




