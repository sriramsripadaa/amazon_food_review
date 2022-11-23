import pandas as pd
from flask import Flask, request, render_template, jsonify
import numpy as np
from sklearn import linear_model
import joblib
import sqlite3
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import flask
app = Flask(__name__)


# replacing some phrases like won't with will not
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

#stop words
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

#function to clean all text reviews
def clean_text(sentance):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # to clean the word of any punctuation or special characters
    sentance = re.sub(r'[?|!|\'|"|#]', r'', sentance)
    sentance = re.sub(r'[.|,|)|(|\|/]', r' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    return sentance.strip()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    review_text = clean_text(to_predict_list['review_text'])
    #test_vect = count_vect.transform(([review_text]))
    pred = clf.predict(count_vect.transform([review_text]))
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    app.run(host='0.0.0.0', port=8080)

"""
con = sqlite3.connect('database.sqlite')
filtered_data = pd.read_sql_query("SELECT * FROM Reviews WHERE Score != 3 LIMIT 10000", con)
score = filtered_data['Score'].apply(lambda x: 1 if x > 3 else 0)

sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final = sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final = final.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]




preprocessed_reviews = []
for sentence in final['Text'].values:
    preprocessed_reviews.append(clean_text(sentence))


vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=10)
vectorizer.fit(preprocessed_reviews)

joblib.dump(vectorizer, 'count_vect.pkl')
X = vectorizer.transform(preprocessed_reviews)
print(X.shape)
Y = final['Score'].values
LR = LogisticRegression(penalty='l2',C=100)
LR.fit(X,Y)
joblib.dump(LR, 'model.pkl')

#print(predict('Have been having this since years. Much better option than Bru.Nescafe still managing to do well in market with all the competitors breathing down it\'s neck. Good one!'))
"""
