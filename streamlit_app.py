import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

# Load the trained XGBoost model and encoders
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
person_gender_encoder = pickle.load(open("gender_encode.pkl", "rb"))
previous_loan_encoder = pickle.load(open("previous_loan_encode.pkl", "rb"))
person_education_encoder = pickle.load(open("person_education_encode.pkl", "rb"))

# Function to preprocess input data
def preprocess_input(user_input):
    # Preprocessing gender
    user_input['person_gender'] = user_input['person_gender'].map({'Male': 1, 'Female': 0})

    # Preprocessing previous_loan_defaults_on_file
    user_input['previous_loan_defaults_on_file'] = user_input['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    
    # Encoding education level
    user_input = user_input.replace({"person_education": person_education_encoder["person_education"]})
    
    # One-hot encode categorical variables like 'person_home_ownership' and 'loan_intent'
    home_ownership_encoder = OneHotEncoder()
    loan_intent_encoder = OneHotEncoder()
    
    # Example: Assuming home_ownership and loan_intent were previously encoded
    person_home_ownership_train = pd.DataFrame(home_ownership_encoder.transform(user_input[['person_home_ownership']]).toarray(), 
                                              columns=home_ownership_encoder.get_feature_names_out())
    loan_intent_train = pd.DataFrame(loan_intent_encoder.transform(user_input[['loan_intent']]).toarray(), 
                                      columns=loan_intent_encoder.get_feature_names_out())
    
    # Add the one-hot encoded columns back to the dataframe
    user_input = pd.concat([user_input, person_home_ownership_train, loan_intent_train], axis=1)
    
    # Drop original columns after encoding
    user_input = user_input.drop(['person_home_ownership', 'loan_intent'], axis=1)

    # Scaling numerical data (assuming RobustScaler was used for features)
    scaler = RobustScaler()
    numerical_columns = user_input.select_dtypes(include=['number']).columns.tolist()
    
    for col in numerical_columns:
        user_input[col] = scaler.fit_transform(user_input[[col]])
    
    return user_input

# Function to make predictions
def make_prediction(input_data):
    processed_input = preprocess_input(input_data)
    prediction = xgb_model.predict(processed_input)
    return prediction

# Streamlit UI
st.title("Loan Status Prediction")
st.write("Please fill out the form below to predict the loan status.")

# Get user input
person_gender = st.selectbox("Gender", ["Male", "Female"])
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["Yes", "No"])
person_education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_home_ownership = st.selectbox("Home Ownership", ["Own", "Mortgage", "Rent"])
loan_intent = st.selectbox("Loan Intent", ["Debt Consolidation", "Home Improvement", "Other"])

# Create a dictionary to store the input
user_input = {
    "person_gender": person_gender,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
    "person_education": person_education,
    "person_home_ownership": person_home_ownership,
    "loan_intent": loan_intent,
}

# When the user clicks the 'Predict' button
if st.button("Predict Loan Status"):
    prediction = make_prediction(pd.DataFrame([user_input]))
    
    if prediction == 1:
        st.write("The loan will likely be **approved**.")
    else:
        st.write("The loan will likely **not be approved**.")
