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


nltk.download('all')


# In[16]:


pip install lightgbm


# In[19]:


pip install rich


# In[25]:


pip install scikit-optimize


# In[3]:


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


# In[4]:


train = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/test.csv')

X_train = train['text'].apply(data_processing)
X_train = X_train.apply(lambda x: stemming(x))

X_test =  test['text'].apply(data_processing)
X_test = X_test.apply(lambda x: stemming(x))

Y_train = train['sentiment']


# In[5]:


def tokenize_text(text):
    tokens = word_tokenize(text)
    
    return tokens

text_series = X_train
token_list = []

for text in text_series:
    tokens = tokenize_text(text)
    token_list.append(tokens)
    
tokens = pd.Series(token_list)


# In[6]:


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


# In[7]:


clean_token_list = []

for token in tokens:
    clean_token = list(filter(lambda x: x not in common_words, token))
    clean_token_list.append(clean_token)

clean_tokens = pd.Series(clean_token_list)


# In[35]:


class_counts = Y_train.value_counts()
class_counts


# In[8]:


## 토큰에 tag붙임. 이거 작업 좀 걸림.

tagged_tokens = clean_tokens.apply(lambda tokens: pos_tag(tokens))


## CD(숫자를 나타내는 품사), NNP(고유명사. 단수형), NNPS(고유명사, 복수형) tag 제거 후 Series객체로 반환

filtered_series = [[word for word, tag in tagged_tokens if tag not in ['CD','NNP','NNPS','PRP','PRP$','DT','IN']] for tagged_tokens in tagged_tokens]
filtered_series = pd.Series(filtered_series)


## 토큰화 되어있는 Series객체 토큰들 합치기

X_train = filtered_series.apply(lambda tokens:' '.join(tokens))


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def get_pipe(model, model_name: str) -> Pipeline:
    "TfidfVectorizer와 모델을 연결한 파이프라인을 반환하는 함수"
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    pipe = Pipeline([
        ("tfidf", tfidf),
        (model_name, model)
    ])
    return pipe


# In[17]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


models = [
    ("nb", BernoulliNB()),
    ("SGD", SGDClassifier(random_state=42, n_jobs=-1)),
    ("logistic", SGDClassifier(loss="log", random_state=42, n_jobs=-1)),
    ("rfc", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ("pac", PassiveAggressiveClassifier(random_state=42, n_jobs=-1)),
    ("lgbm", LGBMClassifier(random_state=42, n_jobs=-1))
]

model_pipes = [(name, get_pipe(model, name)) for name, model in models]


# In[21]:


import rich  # 출력을 이쁘게 꾸며주는 라이브러리
import numpy as np
from tqdm.auto import tqdm
from rich.table import Table
from sklearn.model_selection import cross_val_score

table = Table(title="Model Comparison Table")
table.add_column("Model Name", justify="left", style="green")
table.add_column("Accuracy", justify="right")

for model_name, model in tqdm(model_pipes, leave=False):
    acc = cross_val_score(
        model, X_train, Y_train, scoring="accuracy", n_jobs=-1
    )
    acc = np.mean(acc)
    table.add_row(model_name, f"{acc:0.4f}")

rich.print(table)


# In[33]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def return_kfold_accuarcy(model, k: int = 5) -> float:
    "모델을 입력받아 KFold 예측 후 accuracy score를 반환하는 함수"
    kfold = StratifiedKFold(k, shuffle=True, random_state=42)
    result = []
    for train_idx, test_idx in kfold.split(X_train, Y_train):
        train, val = X_train.iloc[train_idx], Y_train.iloc[train_idx]
        model.fit(train, val)
        pred = model.predict(test)
        acc = accuracy_score(val, pred)
        result.append(acc)

    return np.mean(result)


# In[ ]:


stacking.fit(X_train, Y_train)
submission_pred = stacking.predict(X_test)


# In[ ]:


submission = pd.read_csv("sample_submission.csv")
submission["label"] = submission_pred
submission

