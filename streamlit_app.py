import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and encoders
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))

# Load encoders
gender_encoder = pickle.load(open("gender_score_encode.pkl", "rb"))
loan_intent_encoder = pickle.load(open("loan_intent_encode.pkl", "rb"))
education_encoder = pickle.load(open("person_education_encode.pkl", "rb"))
defaults_encoder = pickle.load(open("previous_loan_encode.pkl", "rb"))

# Load scalers for numerical data
scalers = {col: pickle.load(open(f"{col}_scaler.pkl", "rb")) for col in ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']}

# Streamlit UI
st.title("Loan Cancellation Prediction")

# Input fields
person_age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Select Gender", ["Male", "Female"])
person_education = st.selectbox("Select Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.number_input("Enter Income", min_value=0)
person_emp_exp = st.number_input("Enter Employment Experience", min_value=0)
loan_amnt = st.number_input("Enter Loan Amount", min_value=1000)
loan_int_rate = st.number_input("Enter Interest Rate", min_value=0.0, max_value=100.0)
loan_percent_income = st.number_input("Enter Loan Percent Income", min_value=0.0, max_value=100.0)
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850)
previous_loan_defaults = st.selectbox("Previous Loan Defaults", ["Yes", "No"])
person_home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["DEBT CONSOLIDATION", "EDUCATION", "HOME IMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])

# Convert categorical inputs using the loaded encoders
person_gender = gender_encoder.transform([[person_gender]]).toarray().flatten()
loan_intent = loan_intent_encoder.transform([[loan_intent]]).toarray().flatten()
person_education = education_encoder.transform([[person_education]]).toarray().flatten()
previous_loan_defaults = defaults_encoder.transform([[previous_loan_defaults]]).toarray().flatten()

# Create input data frame
input_data = pd.DataFrame([{
    'person_age': person_age,
    'person_gender': person_gender[0],  # Assuming binary encoding
    'person_education': person_education[0],  # Assuming one-hot encoding
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': previous_loan_defaults[0]  # Assuming binary encoding
}])

# Scale numerical features
for col in ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']:
    input_data[[col]] = scalers[col].transform(input_data[[col]])

# One-hot encoding for categorical columns if needed
# No need for manual one-hot encoding since it's handled by the encoders

# Predict using the trained XGBoost model
prediction = xgb_model.predict(input_data)
prediction_proba = xgb_model.predict_proba(input_data)

# Display the prediction and probability
if prediction[0] == 1:
    st.subheader("Prediction: Loan Cancellation Likely")
    st.write(f"Probability: {prediction_proba[0][1]:.2f}")
else:
    st.subheader("Prediction: Loan Cancellation Unlikely")
    st.write(f"Probability: {prediction_proba[0][0]:.2f}")
