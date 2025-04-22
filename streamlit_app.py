import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Load pre-trained models and encoders
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
gender_encoder = pickle.load(open('gender_encode.pkl', 'rb'))
previous_loan_encoder = pickle.load(open('previous_loan_encode.pkl', 'rb'))
person_education_encoder = pickle.load(open('person_education_encode.pkl', 'rb'))
loan_intent_encoder = pickle.load(open('loan_intent_encode.pkl', 'rb'))

# Load scaler for feature scaling
scalers = {}
for col in ['person_income']:  # List of numerical columns that have been scaled
    scalers[col] = pickle.load(open(f"{col}_scaler.pkl", "rb"))

# Function for preprocessing user input
def preprocess_user_input(user_input):
    # Gender
    user_input['person_gender'] = gender_encoder.get(user_input['person_gender'], 0)  # Default to 0 if not found

    # Previous loan default
    user_input['previous_loan_defaults_on_file'] = previous_loan_encoder.get(user_input['previous_loan_defaults_on_file'], 0)

    # Education level
    user_input = person_education_encoder.get(user_input['person_education'], 0)

    # Loan intent (OneHotEncoder)
    loan_intent_encoded = loan_intent_encoder.transform([[user_input['loan_intent']]]).toarray()
    loan_intent_columns = loan_intent_encoder.get_feature_names_out()

    # Creating DataFrame for OneHotEncoded features
    loan_intent_df = pd.DataFrame(loan_intent_encoded, columns=loan_intent_columns)

    # Merge the loan intent OneHotEncoded columns with user input DataFrame
    user_input = pd.DataFrame([user_input])
    user_input = pd.concat([user_input, loan_intent_df], axis=1)

    # Scale numerical values
    for col in ['person_income']:
        user_input[[col]] = scalers[col].transform(user_input[[col]])

    return user_input

# Main Streamlit App
def main():
    st.title("Loan Status Prediction")

    # Input fields for user
    person_gender = st.selectbox("Select Gender", ["Male", "Female"])
    person_income = st.number_input("Enter Income", min_value=0)
    person_education = st.selectbox("Select Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])
    loan_intent = st.selectbox("Loan Intent", ["DEBT CONSOLIDATION", "HOME IMPROVEMENT", "PERSONAL LOAN", "OTHER"])

    # Create a dictionary to hold user input
    user_input = {
        "person_gender": person_gender,
        "person_income": person_income,
        "person_education": person_education,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
        "loan_intent": loan_intent,
    }

    # Button to trigger prediction
    if st.button("Predict"):
        # Preprocess user input
        processed_input = preprocess_user_input(user_input)

        # Make prediction using the XGBoost model
        prediction = xgb_model.predict(processed_input)
        
        # Display the prediction result
        if prediction == 1:
            st.success("The loan status prediction: Approved")
        else:
            st.error("The loan status prediction: Denied")

if __name__ == "__main__":
    main()
