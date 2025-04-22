import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load all encoders & model
gender_encoder = pickle.load(open("gender_encode.pkl", "rb"))
education_encoder = pickle.load(open("person_education_encode.pkl", "rb"))
previous_loan_encoder = pickle.load(open("previous_loan_encode.pkl", "rb"))
loan_intent_encoder = pickle.load(open("loan_intent_encode.pkl", "rb"))
model = pickle.load(open("xgb_model.pkl", "rb"))

# Load all scalers
scaler_files = {
    "person_age": "person_age_scaler.pkl",
    "person_income": "person_income_scaler.pkl",
    "person_emp_length": "person_emp_length_scaler.pkl",
    "loan_amnt": "loan_amnt_scaler.pkl",
    "loan_int_rate": "loan_int_rate_scaler.pkl"
}
scalers = {col: pickle.load(open(path, "rb")) for col, path in scaler_files.items()}

# Function to preprocess single row input
def preprocess_input(df):
    df = df.copy()

    # Gender encoding
    df['person_gender'] = df['person_gender'].map(gender_encoder)

    # Education encoding
    df['person_education'] = df['person_education'].map(education_encoder['person_education'])

    # Previous loan default
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(previous_loan_encoder)

    # OneHot loan_intent
    loan_intent_ohe = loan_intent_encoder.transform(df[['loan_intent']]).toarray()
    loan_intent_df = pd.DataFrame(loan_intent_ohe, columns=loan_intent_encoder.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), loan_intent_df.reset_index(drop=True)], axis=1)
    df.drop('loan_intent', axis=1, inplace=True)

    # Drop other unneeded columns if any
    if 'loan_status' in df.columns:
        df.drop('loan_status', axis=1, inplace=True)

    # Scaling numeric columns
    for col in scalers:
        if col in df.columns:
            df[col] = scalers[col].transform(df[[col]])

    return df

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Loan Default Prediction App")

    uploaded_file = st.file_uploader("📄 Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Original Data Preview")
        st.write(df.head())

        # Preprocess
        try:
            X_processed = preprocess_input(df)
        except Exception as e:
            st.error(f"❌ Error saat preprocessing data: {e}")
            return

        # Predict
        preds = model.predict(X_processed)
        df['prediction'] = preds
        st.subheader("📈 Prediction Result")
        st.write(df[['prediction']])

        # Option to display class distribution
        st.subheader("📊 Class Distribution")
        st.bar_chart(df['prediction'].value_counts())

        # Optional: Download result
        csv_result = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", data=csv_result, file_name="loan_prediction_result.csv", mime="text/csv")

    else:
        st.info("Silakan unggah file CSV berisi data pinjaman untuk diprediksi.")

if __name__ == "__main__":
    main()
