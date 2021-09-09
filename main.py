from flask import Flask, render_template
from flask.globals import request
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('train_matrix.pkl')
vectorizer = joblib.load('cvector.pkl')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    string = request.form.get("string")
    document = [string]
    sparse_matrix = vectorizer.transform(document)
    output = cosine_similarity(model, sparse_matrix)
    return render_template('index.html', prediction_text = output)

app.run(debug=True)