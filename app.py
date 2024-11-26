from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open("model/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")  # Render the input form

@app.route('/predict', methods=["POST"])
def predict():
    # Extract data from the form
    try:
        gender = int(request.form["gender"])
        married = int(request.form["married"])
        dependents = int(request.form["dependents"])
        education = int(request.form["education"])
        self_employed = int(request.form["self_employed"])
        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        property_area = int(request.form["property_area"])
        
        # Combine inputs into a numpy array
        input_data = np.array([gender, married, dependents, education, self_employed,
                               applicant_income, coapplicant_income, loan_amount, loan_term,
                               property_area]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        result = "Good Credit (Loan Likely Approved)" if prediction[0] == 1 else "Bad Credit (Loan Likely Denied)"
        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
