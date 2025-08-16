import sys
import os
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import traceback

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # FIXED: Get 'ethnicity' from form (not 'race_ethnicity')
            # because your HTML form field is named 'ethnicity'
            data = CustomData(
                gender=request.form.get('gender'),
                ethnicity=request.form.get('ethnicity'),  # ← Fixed: get 'ethnicity' from form
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_data_frame()
            print("DataFrame created:")
            print(pred_df)
            print(f"Columns: {pred_df.columns.tolist()}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print(f"Results: {results}")
            
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            print(f"Error in Flask route: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 

