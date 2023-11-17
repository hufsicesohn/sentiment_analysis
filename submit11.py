# 데이터 전처리 부분 수정, 랜덤 포레스트 사용
import pandas as pd
import numpy as np
import matplotlib as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag

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

# 토큰으로 변환
def tokenize_text(text):
    tokens = word_tokenize(text)
    
    return tokens

text_series = X_train
token_list = []

# 문장을 단어 토큰으로 변환
for text in text_series:
    tokens = tokenize_text(text)
    token_list.append(tokens)
    
tokens = pd.Series(token_list)

# 각 class의 단어들의 빈도수 구하기
def get_freq_count(tokens):
    neg_token = tokens[train['sentiment'] == 2]
    pos_token = tokens[train['sentiment'] == 1]
    neul_token = tokens[train['sentiment'] == 0]
    neg_freq = pd.Series(np.concatenate([w for w in neg_token])).value_counts()
    pos_freq = pd.Series(np.concatenate([w for w in pos_token])).value_counts()
    neul_freq = pd.Series(np.concatenate([w for w in neul_token])).value_counts()
    
    return neg_freq, pos_freq, neul_freq

neg_freq ,pos_freq, neul_freq = get_freq_count(tokens)

# 각 class의 빈도수가 가장 많은 단어 3개 추출
top_50_pos = pos_freq[:3]
top_50_neg = neg_freq[:3]
top_50_neul = neul_freq[:3]

# 세 개의 class에 겹치는 상위 3개의 단어들은 제거
common_words = [p for p in top_50_neg.index if p in set.intersection(set(top_50_pos.index), set(top_50_neul.index))]

clean_token_list = []

for token in tokens:
    clean_token = list(filter(lambda x: x not in common_words, token))
    clean_token_list.append(clean_token)

clean_tokens = pd.Series(clean_token_list)


# 토큰에 품사 tag 붙이는 작업
tagged_tokens = clean_tokens.apply(lambda tokens: pos_tag(tokens))


# CD(숫자를 나타내는 품사), NNP(고유명사. 단수형), NNPS(고유명사, 복수형) tag 제거 후 Series객체로 반환
filtered_series = [[word for word, tag in tagged_tokens if tag not in ['CD','NNP','NNPS']] for tagged_tokens in tagged_tokens]
filtered_series = pd.Series(filtered_series)


# 토큰화 되어있는 Series객체 토큰들 합치기
X_train = filtered_series.apply(lambda tokens:' '.join(tokens))

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,2)).fit(x_train)
x_train_vect = vect.transform(x_train)

clf = RandomForestClassifier(n_estimators=500,min_samples_split=10)
clf.fit(x_train_vect, y_train)

X_test = vect.transform(X_test)
preds = log.predict(X_test)
submit11 = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit11['sentiment'] = preds
submit11.head()

submit11.to_csv('C:/Users/sohnp/baseline_submit14_심규상.csv', index=False)
print('Done')



