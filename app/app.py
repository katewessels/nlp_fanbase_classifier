from flask import Flask, request, render_template
import numpy as np
import pickle
from random import random

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('starter_template.html')

# about page
@app.route('/about/')
def about():
    return render_template('starter_template2.html')

# contact page
@app.route('/contact/')
def contact():
    return render_template('starter_template3.html')



# Form page to submit text
# @app.route('/')
# def submission_page():
#     return render_template('index.html')

# My prediction app
###Modify this for my data
@app.route('/prediction', methods=['POST'] )
def predict():
    with open("model.pkl", 'rb') as f:
        model = pickle.load(f)
        X = np.array([request.form['sepal_length'],
                      request.form['sepal_width'],
                      request.form['petal_length'],
                      request.form['petal_width']]).astype(float).reshape(1, -1)
        probs = model.predict_proba(X)
    page = f'''Predicted probabilities:
    <table>
        <tr><th>species</th><th>probability</th></tr>
        <tr><td>iris setosa</td><td>{probs[0][0]:.2f}</td></tr>
        <tr><td>iris versicolor</td><td>{probs[0][1]:.2f}</td></tr>
        <tr><td>iris virginica</td><td>{probs[0][2]:.2f}</td></tr>
    </table>'''

    return page


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
