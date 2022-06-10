# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load(r"D:\sumit_DS project\end to end ML project\salary_predictor\salary_predictor.pkl")


df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index1.html', prediction_text='Please enter valid Experience')
        

    output = model.predict([features_value])[0][0].round(2)

    # input and predicted value store in df then save in csv file
    df= pd.concat([df,pd.DataFrame({'Experience':input_features,'Predicted Output':[output]})],ignore_index=True)
    print(df)   
    df.to_csv('sal_data_from_app.csv')

    return render_template('index1.html', prediction_text='Your Salary is {} INR, when you have {} years of Experience '.format(output, int(features_value[0])))



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)

#if __name__ == "__main__":
#    app.run(host='0.0.0.0', port=8080)
    
