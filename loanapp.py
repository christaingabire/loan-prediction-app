import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.title("Loan Approval Predictor")

# Input form
st.header("Enter Applicant Details:")
gender = st.selectbox("Gender", options=["Male", "Female"])
married = st.selectbox("Married", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

# Convert inputs to model format
if st.button("Predict"):
    # Encode inputs
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = int(dependents[0]) if dependents != "3+" else 3
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 1, "Semiurban": 0, "Rural": 2}[property_area]

    # Create input array
    input_data = np.array([gender, married, dependents, education, self_employed,
                           applicant_income, coapplicant_income, loan_amount,
                           loan_term, property_area]).reshape(1, -1)

    # Predict
    prediction = model.predict(input_data)
    result = "Good Credit (Loan Likely Approved)" if prediction[0] == 1 else "Bad Credit (Loan Likely Denied)"
    st.success(f"Prediction: {result}")
