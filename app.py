from flask import Flask
from flask import Flask, request, jsonify, render_template
from joblib import dump, load
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
import pickle


app = Flask(__name__)
model = load('model.joblib') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    comment = [x for x in request.form.values()][0]
    
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("vectorizer.pickle", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array([comment])))

    # print('=====',comment)
    rating = model.predict(tfidf)[0]
    # comment2 = count_vectorizer.transform([comment])


    return render_template('index.html', prediction_text='Predicted Rating should be {} stars'.format(rating))
    # final_features = [np.array(int_features)]

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

