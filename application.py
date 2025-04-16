import pickle
import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# import Linear Regression and standard scaler pickle
regression_model=pickle.load(open("regression.pkl","rb"))
standard_scaler=pickle.load(open("scaler.pkl","rb"))



@app.route('/')
def index():
    return render_template("index.html")
@app.route("/predictdata",methods=['GET','POST'])
def predict_data():
    if request.method=="POST":
        Weight= float(request.form.get("Weight"))
        new_data=standard_scaler.transform([[Weight]])
        result=regression_model.predict(new_data)
        return render_template("home.html",results=result[0])


        
    else:
        return render_template("home.html")


    

if __name__ == '__main__':
   app.run(debug=True)
