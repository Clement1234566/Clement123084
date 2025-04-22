import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Function to preprocess the input data (using encoder and model)
def preprocess_input(user_input):
    # Assuming the encoder is a LabelEncoder, for example
    # If you used OneHotEncoder, you'll need to adapt this
    processed_data = user_input.copy()

    # Example: If the 'hotel_type' column was encoded
    processed_data['hotel_type'] = encoder.transform([user_input['hotel_type']])[0]  # Use encoder to transform the input

    # Convert the processed_data to a DataFrame
    input_df = pd.DataFrame([processed_data])

    return input_df

# Function to make a prediction
def make_prediction(input_data):
    # Preprocess the data
    processed_input = preprocess_input(input_data)

    # Use the model to predict
    prediction = model.predict(processed_input)

    return prediction

# Streamlit UI
st.title("Hotel Booking Cancellation Prediction")

st.write("Please fill out the form below to predict whether a booking will be cancelled or not.")

# Get user input
hotel_type = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=10)
adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
babies = st.number_input("Number of Babies", min_value=0, max_value=10, value=0)
meal = st.selectbox("Meal Type", ["BB", "HB", "FB", "SC", "Undefined"])
country = st.text_input("Country", "USA")
market_segment = st.selectbox("Market Segment", ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Undefined"])

# Create a dictionary to store the input
user_input = {
    "hotel_type": hotel_type,
    "lead_time": lead_time,
    "adults": adults,
    "children": children,
    "babies": babies,
    "meal": meal,
    "country": country,
    "market_segment": market_segment
}

# When the user clicks the 'Predict' button
if st.button("Predict Cancellation"):
    prediction = make_prediction(user_input)
    
    if prediction == 1:
        st.write("The booking will likely be **canceled**.")
    else:
        st.write("The booking will likely **not be canceled**.")
