
import pandas as pd
import numpy as np
import matplotlib as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

train = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/train.csv')
test = pd.read_csv('C:/Users/sohnp/Downloads/open (2)/test.csv')

Train = train['text']
Test = test['text']

# 데이터 전처리 부분에 표제어 추출하는 작업 추가
def data_processing(text):
    text = re.sub(r"https\S+|www|\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    text = text.lower().split()
    stops = set(stopwords.words('english'))
    text = [word for word in text if not word in stops]

    # 텍스트 내 단어들을 표제어(기본 사전형태)로 변환
    lemmatizer = nltk.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    
    return " ".join(text)


X_train = train['text'].apply(data_processing)
X_test =  test['text'].apply(data_processing)

Y_train = train['sentiment']

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

pipeline = Pipeline([
                     ('cnt_vect', CountVectorizer(stop_words = 'english', ngram_range=(1, 2))), 
                     ('lr_clf', LogisticRegression(multi_class='multinomial', solver='lbfgs',C=10))])

# Pipeline 객체를 이용해 fit(), predict()로 학습/예측 수행, predict_proba()는 roc_auc 때문에 수행. 
pipeline.fit(X_train, Y_train)
pred = pipeline.predict(X_test)

submit10 = pd.read_csv('C:/Users/sohnp/Downloads/open (1)/sample_submission.csv')
submit10['sentiment'] = pred
submit10.head()

submit10.to_csv('C:/Users/sohnp/baseline_submit14.csv', index=False)
print('Done')



