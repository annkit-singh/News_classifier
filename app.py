from flask import Flask,render_template,url_for,request
import numpy as np
import os
import flask
import newspaper
from flask_cors import CORS
from newspaper import Article
import urllib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

NB_model = open('NBmodel.pkl','rb')
model = joblib.load(NB_model)


@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        str = request.form.get("inputnews") 
        vect=loaded_vectorizer.transform(  [str]  )
        vect=vect.toarray()
        my_prediction = model.predict(vect)
        if(my_prediction==1):
            out="True"
        else:
            out="False"
    return render_template('main.html',prediction_text='The news is "{}"'.format(out))
    


if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)