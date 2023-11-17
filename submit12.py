# negative class의 데이터가 압도적으로 많아 세 개의 class에 겹치는 상위 3개의 단어들을 negative class에서만 지우기
# smote를 이용하여 오버샘플링된 데이터의 양을 균형있게 조절
# 앙상블 학습 기법 중 하나인 투표 기반 분류기 구현, 랜덤 포레스트와 그래디언트 부스팅 두 가지 모델 조합
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

train = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/test.csv')

X_train = train['text'].apply(data_processing)
X_train = X_train.apply(lambda x: stemming(x))

X_test =  test['text'].apply(data_processing)
X_test = X_test.apply(lambda x: stemming(x))

Y_train = train['sentiment']

def tokenize_text(text):
    tokens = word_tokenize(text)
    
    return tokens

text_series = X_train
token_list = []

for text in text_series:
    tokens = tokenize_text(text)
    token_list.append(tokens)
    
tokens = pd.Series(token_list)

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

tagged_tokens = clean_tokens.apply(lambda tokens: pos_tag(tokens))

filtered_series = [[word for word, tag in tagged_tokens if tag not in ['CD','NNP','NNPS','PRP','PRP$','DT','IN']] for tagged_tokens in tagged_tokens]
filtered_series = pd.Series(filtered_series)

X_train = filtered_series.apply(lambda tokens:' '.join(tokens))

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
submit12.to_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment_test.csv', index=False)

print('Done')

submit11 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment_test.csv')



