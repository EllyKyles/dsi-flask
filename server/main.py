from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('../models/model.p','rb'))

cv = pickle.load(open('../models/cv.p','rb'))

@app.route('/hello',methods=['GET'])
def hello():
    return "hello world"

@app.route('/add/<int:a>/<int:b>', methods=['GET'])
def add(a,b):
    return "the sum of {} and {} is {}".format(a,b,a+b) 


@app.route('/funny', methods=['GET'])
def funny():
    return render_template('fun.html')


@app.route('/product', methods=['GET','POST'])
def product():
    if request.method=="POST":
        data= request.form['nums']
        nums =[float(s) for s in data.split(",")]
        prod = np.product(nums)
        return render_template('product.html', prod=prod)
    else:
        print("method:", request.method)
        return render_template('product.html')


@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    return render_template('diagnostics.html', model=model)


@app.route('/sentiment', methods=['GET','POST'])
def sentiment():
    if request.method=='POST':
        review = request.form['review']
        vector = cv.transform([review])
        prediction = model.predict_proba(vector)
        c0,c1 = prediction[0]
        print('review:',review, 'vecotr:',vector.shape, 'predict:',prediction)
        return render_template("sentiment.html", model=model,c0=c0.round(3),c1=c1.round(3), review=review)
        
    else:
        return render_template('sentiment.html', model=model)