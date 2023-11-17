# 전처리 작업, 베르누이 나이브 베이즈 분류 사용
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB
import re
import nltk

train = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']

# 전처리 작업
def preprocess(x):
  # 불필요한 문자 삭제
  for i in range(32000):
    x[i] = re.sub('[^a-zA-Z]', ' ', x[i])

  # 모든 단어 소문자로 변환
  for i in range(32000):
    x[i] = x[i].lower().split()

  # 불용어 제거
  for i in range(32000):
    stops = set(stopwords.words('english'))
    x[i] = [word for word in x[i] if not word in stops]

  # 단어의 어간 추출
  for i in range(32000):
    stemmer = nltk.stem.SnowballStemmer('english')
    x[i] = [stemmer.stem(word) for word in x[i]]

  for i in range(32000):
    x[i] = ' '.join(x[i])

  return x

X_train = preprocess(X_train)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

X_test = preprocess(X_test)
X_test = vectorizer.transform(X_test).toarray()
preds = bnb.predict(X_test)

submit5 = pd.read_csv('/sample_submission.csv')
submit5['sentiment'] = preds
submit5.head()

submit5.to_csv('C:/Users/sohnp/baseline_submit9.csv', index=False)
print('Done')

submit5 = pd.read_csv('C:/Users/sohnp/baseline_submit9.csv')

