import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle


# Load the dataset
data = pd.read_excel("data/LoanPrediction-Kaggle dataset.xlsx", header=1)

# Preview the first few rows
print(data.head())

# Remove missing values
data.dropna(inplace=True)

# Convert categorical data to numeric
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 1, 'Semiurban': 0, 'Rural': 2})

# Heatmap
sns.countplot(x='Credit_History', data=data)
plt.title('Loan Approval Distribution')
plt.show()

# Distribution of Applicant Income
sns.histplot(data['ApplicantIncome'], kde=True, bins=20)
plt.title('Distribution of Applicant Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Loan Amount vs. Loan Status
sns.boxplot(x='Credit_History', y='LoanAmount', data=data)
plt.title('Loan Amount vs Loan Status')
plt.xlabel('Credit_History (1: Approved, 0: Denied)')
plt.ylabel('Loan Amount')
plt.show()

# Train-Test Split
X = data.drop(columns=['Loan_ID', 'Credit_History'])  # Drop Loan_ID and target
y = data['Credit_History']  # Target

# Balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Use balanced data for training
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Feature correlation heatmap
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
corr = numeric_data.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title('Feature Correlation')
plt.show()


#save trained model to a file
with open("model/loan_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model saved successfully")