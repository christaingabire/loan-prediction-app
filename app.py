from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import fcntl


#initialize Flask app
app = Flask(__name__)

#Load the trained model
with open("model/loan_model.pkl", "rb") as file:
    model = pickle.load(file)
    
#Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

#Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    #Get from data
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    dependents = int(request.form['dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_amount_term = float(request.form['loan_amount_term'])
    property_area = int(request.form['property_area'])
    
    #prepare data for prediction
    features = np.array([[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, property_area]])
    
    #make prediction
    prediction = model.predict(features)
    
    #interpret the prediction
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    
    return render_template('index.html', prediction_text =f'Loan Status: {result}')

if __name__ == '_main_':
    #use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host= '0.0.0.0', port=port)