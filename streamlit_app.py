import streamlit as st
import pandas as pd
import pickle
from models.data_preprocessor import DataPreprocessor
from models.model_trainer import ModelTrainer
from models.model_saver import ModelSaver
from utils.encoders import Encoders
from utils.scalers import Scalers

# Load pre-trained model and encoders
xgb_model = pickle.load(open('assets/xgb_model.pkl', 'rb'))

def preprocess_user_input(user_input):
    # Apply encoding and scaling based on user input
    user_input['person_gender'] = Encoders.encode_gender(user_input['person_gender'])
    user_input['previous_loan_defaults_on_file'] = Encoders.encode_previous_loan(user_input['previous_loan_defaults_on_file'])
    user_input['person_education'] = Encoders.encode_education(user_input['person_education'])
    user_input['loan_intent'] = Encoders.encode_loan_intent(user_input['loan_intent'])

    # Scale numerical columns
    user_input['person_income'] = Scalers.scale_feature('person_income', user_input['person_income'])

    return pd.DataFrame([user_input])

def main():
    st.title("Loan Status Prediction")

    # Input fields for user
    person_gender = st.selectbox("Select Gender", ["Male", "Female"])
    person_income = st.number_input("Enter Income", min_value=0)
    person_education_text = st.selectbox("Select Education", list(person_education_encoder.keys()))
    person_education = person_education_encoder[person_education_text]

    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])
    loan_intent = st.selectbox("Loan Intent", ["DEBT CONSOLIDATION", "HOME IMPROVEMENT", "PERSONAL LOAN", "OTHER"])

    user_input = {
        "person_gender": person_gender,
        "person_income": person_income,
        "person_education": person_education,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
        "loan_intent": loan_intent,
    }

    if st.button("Predict"):
        processed_input = preprocess_user_input(user_input)
        prediction = xgb_model.predict(processed_input)
        
        if prediction == 1:
            st.success("The loan status prediction: Approved")
        else:
            st.error("The loan status prediction: Denied")

if __name__ == "__main__":
    main()
