import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and encoders
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
home_ownership_encoder = pickle.load(open("home_ownership_encoder.pkl", "rb"))
loan_intent_encoder = pickle.load(open("loan_intent_encoder.pkl", "rb"))
scalers = {col: pickle.load(open(f"{col}_scaler.pkl", "rb")) for col in ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']}

# Maps for encoding
education_map = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
gender_map = {"Male": 1, "Female": 0}
defaults_map = {"Yes": 1, "No": 0}

# Streamlit UI
st.title("Loan Cancellation Prediction")

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

# Convert user inputs to correct format
person_gender = gender_map[person_gender]
person_education = education_map[person_education]
previous_loan_defaults = defaults_map[previous_loan_defaults]

# One-hot encoding for categorical columns
home_ownership_ohe = home_ownership_encoder.transform([[person_home_ownership]]).toarray()
loan_intent_ohe = loan_intent_encoder.transform([[loan_intent]]).toarray()

# Create input data frame
input_data = pd.DataFrame([{
    'person_age': person_age,
    'person_gender': person_gender,
    'person_education': person_education,
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': previous_loan_defaults
}])

# Scale numerical features
for col in ['person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']:
    input_data[[col]] = scalers[col].transform(input_data[[col]])

# Combine one-hot encoded columns with input data
ohe_df = pd.DataFrame(
    np.hstack([home_ownership_ohe, loan_intent_ohe]),
    columns = list(home_ownership_encoder.get_feature_names_out()) + list(loan_intent_encoder.get_feature_names_out())
)
final_input = pd.concat([input_data.reset_index(drop=True), ohe_df], axis=1)

# Predict using the trained XGBoost model
prediction = xgb_model.predict(final_input)
prediction_proba = xgb_model.predict_proba(final_input)

# Display the prediction and probability
if prediction[0] == 1:
    st.subheader("Prediction: Loan Cancellation Likely")
    st.write(f"Probability: {prediction_proba[0][1]:.2f}")
else:
    st.subheader("Prediction: Loan Cancellation Unlikely")
    st.write(f"Probability: {prediction_proba[0][0]:.2f}")
