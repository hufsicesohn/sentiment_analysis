# 랜덤포레스트에 각 class에 대한 가중치 적용
# 필터링할 품사 태그 추가
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

def get_freq_count(tokens):
    neg_token = tokens[train['sentiment'] == 2]
    pos_token = tokens[train['sentiment'] == 1]
    neul_token = tokens[train['sentiment'] == 0]
    neg_freq = pd.Series(np.concatenate([w for w in neg_token])).value_counts()
    pos_freq = pd.Series(np.concatenate([w for w in pos_token])).value_counts()
    neul_freq = pd.Series(np.concatenate([w for w in neul_token])).value_counts()
    
    return neg_freq, pos_freq, neul_freq


neg_freq ,pos_freq, neul_freq = get_freq_count(tokens)

top_50_pos = pos_freq[:3]
top_50_neg = neg_freq[:3]
top_50_neul = neul_freq[:3]

common_words = [p for p in top_50_neg.index if p in set.intersection(set(top_50_pos.index), set(top_50_neul.index))]

clean_token_list = []

for token in tokens:
    clean_token = list(filter(lambda x: x not in common_words, token))
    clean_token_list.append(clean_token)

clean_tokens = pd.Series(clean_token_list)

tagged_tokens = clean_tokens.apply(lambda tokens: pos_tag(tokens))


# 품사 tag추가하여 불필요한 모든 품사의 단어들 제거
filtered_series = [[word for word, tag in tagged_tokens if tag not in ['CD','NNP','NNPS','PRP','PRP$','DT','IN']] for tagged_tokens in tagged_tokens]
filtered_series = pd.Series(filtered_series)

X_train = filtered_series.apply(lambda tokens:' '.join(tokens))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 랜덤 포레스트 분류기, 가중치 추가(negative class에 대한 가중치 최대)
weights = {0:3.0, 1:2.0, 2:1.0}
rf_classifier = RandomForestClassifier(n_estimators=1000, class_weight=weights, random_state=47)

# 그래디언트 부스팅 분류기
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=47)


ensemble_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('gb', gb_classifier)
], voting='hard')


ensemble_classifier.fit(X_train_vec, Y_train)

Y_test_pred = ensemble_classifier.predict(X_test_vec)

# 결과 저장
submit13 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/sample_submission.csv')
submit13['sentiment'] = Y_test_pred
submit13.to_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment8.csv', index=False)

print('Done')

submit13 = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/ensemble_sentiment8.csv')




