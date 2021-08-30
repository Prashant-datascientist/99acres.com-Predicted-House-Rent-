# Import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from pickle import load
import catboost

app = Flask(__name__)

Cat_Boost = load(open('cat_tune_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    city =request.form.get('city')
    BHKS = request.form.get('BHKS')
    sqft_per_inch = request.form.get('sqft_per_inch')
    build_up_area = request.form.get('build_up_area')
    Type_of_property = request.form.get('Type_of_property')
    location_of_the_property = request.form.get('location_of_the_property')
    deposit = request.form.get('deposit')

     
    #area_sqft = int(area_sqft)
    #area_sqft = np.log(area_sqft)
    print(city)
    print(BHKS)
    print(sqft_per_inch)
    print(build_up_area)
    print(Type_of_property)
    print(location_of_the_property)
    print(deposit)
    #print(bathrooms)
   #print(area_sqft)

    predictions = Cat_Boost.predict([[city,BHKS,sqft_per_inch,build_up_area,Type_of_property,location_of_the_property,deposit]])
    
    predictions = np.exp(predictions)
    

    prediction_text = 'house rant is predicted to be :  '+str(predictions)


    return render_template('index.html', prediction_text=prediction_text)


# Allow the Flask app to launch from the command line
if __name__ == "__main__":
    app.run(debug=True)
