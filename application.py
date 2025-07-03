"""
This file is same file app.py but we creating this file to deploy our application on the cloude
or any cloud platform, we saw the defoult starting point in the code for delpoyement on the 
cloude is application.py in the code in documentation of deploying code in cloud as given below.

option_settings:
   "aws:elasticbeanstalk:container:pyhton":
   WSGIPath: application:application

so, we just coppy the code from app.py to application.py and we will use this file for deployement.
"""

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

## Rout for the home page
@app.route('/')
def name():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        data =  CustomData(
            # reading all the values enter by user for doing prediction via 'GET' method
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get("race_ethnicity"),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            lunch = request.form.get("lunch"),
            test_preparation_course = request.form.get("test_preparation_course"),
            reading_score = float(request.form.get("reading_score")),
            writing_score = float(request.form.get("writing_score"))



        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001)

