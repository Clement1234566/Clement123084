import streamlit as st
import pandas as pd
import pickle
import os

# LOAD MODEL
model = pickle.load(open("xgb_model.pkl", "rb"))

# LOAD ENCODERS
gender_encoder = pickle.load(open("gender_encode.pkl", "rb"))
education_encoder = pickle.load(open("person_education_encode.pkl", "rb"))["person_education"]
loan_encoder = pickle.load(open("loan_intent_encode.pkl", "rb"))
prev_loan_encoder = pickle.load(open("previous_loan_encode.pkl", "rb"))

# LOAD SCALERS
scalers = {}
for col in ["person_income", "person_age", "loan_amnt", "loan_percent_income"]:
    with open(f"{col}_scaler.pkl", "rb") as f:
        scalers[col] = pickle.load(f)

st.title("Loan Cancellation Prediction App")

# Input fields
person_gender = st.selectbox("Select Gender", ["Male", "Female"])
person_income = st.number_input("Enter Income", min_value=0)
person_education = st.selectbox("Select Education", list(education_encoder.keys()))
previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", ["Yes", "No"])
loan_intent = st.selectbox("Loan Intent", ["DEBT CONSOLIDATION", "HOME IMPROVEMENT", "MEDICAL", "VENTURE", "EDUCATION", "PERSONAL"])
person_age = st.number_input("Age", min_value=18)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0)

if st.button("Predict"):
    # Encoding
    gender_val = gender_encoder[person_gender]
    education_val = education_encoder[person_education]
    prev_loan_val = prev_loan_encoder[previous_loan_defaults_on_file]

    # OHE for loan_intent
    intent_df = pd.DataFrame([[loan_intent]], columns=["loan_intent"])
    intent_ohe = loan_encoder.transform(intent_df).toarray()
    intent_columns = loan_encoder.get_feature_names_out()

    # Final input
    input_dict = {
        "person_gender": gender_val,
        "person_income": scalers["person_income"].transform([[person_income]])[0][0],
        "person_education": education_val,
        "previous_loan_defaults_on_file": prev_loan_val,
        "person_age": scalers["person_age"].transform([[person_age]])[0][0],
        "loan_amount": scalers["loan_amount"].transform([[loan_amount]])[0][0],
        "loan_percent_income": scalers["loan_percent_income"].transform([[loan_percent_income]])[0][0],
    }
    
    base_df = pd.DataFrame([input_dict])
    intent_df = pd.DataFrame(intent_ohe, columns=intent_columns)
    final_input = pd.concat([base_df, intent_df], axis=1)

    prediction = model.predict(final_input)[0]
    st.success(f"Prediction: {'Loan Will Be Approved' if prediction == 0 else 'Loan Will Be Cancelled'}")
