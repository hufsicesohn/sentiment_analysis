# 데이터 전처리 함수 수정
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

# 데이터 전처리 한 번에 하는 함수 생성
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

X_train = train['text']
Y_train = train['sentiment']

X_test = test['text']

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
                     ('cnt_vect', CountVectorizer(stop_words = 'english', ngram_range=(1, 2))), 
                     ('lr_clf', LogisticRegression(C=10))])

# Pipeline 객체를 이용해 fit(), predict()로 학습/예측 수행, predict_proba()는 roc_auc 때문에 수행. 
pipeline.fit(x_train, y_train)
pred = pipeline.predict(x_test)

submit9['sentiment'] = pred
submit9.head()
submit9.to_csv('./baseline_submit13.csv', index=False)
print('Done')

submit9 = pd.read_csv('/baseline_submit13.csv')

