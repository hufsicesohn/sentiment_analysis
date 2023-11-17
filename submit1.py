# 가우시안 나이브 베이즈 분류 사용
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/test.csv')

X_train = train['text']
y_train = train['sentiment']

X_test = test['text']

# tdifvectorizer로 단어를 벡터로 변환
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()

# 가우시안 나이브베이즈 분류 학습
classifier = GaussianNB()
classifier.fit(X_train, y_train)

X_test = vectorizer.transform(X_test).toarray()
preds = classifier.predict(X_test)
submit = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit['sentiment'] = preds
submit.head()


submit.to_csv('C:/Users/sohnp/baseline_submit5.csv', index=False)
print('Done')

submit = pd.read_csv('C:/Users/sohnp/baseline_submit5.csv')

