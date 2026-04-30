import streamlit as st
import pickle
import pandas as pd
from utils import load_data

# Load model
model = pickle.load(open("saved_model/model.pkl", "rb"))

st.title("💰 AI Personal Finance Advisor")

# User input
date = st.date_input("Select Date")
category = st.selectbox("Category", ["Food", "Travel", "Shopping", "Bills"])

# Convert input to dataframe
input_data = pd.DataFrame({
    'date': [date],
    'category': [category]
})

# Feature engineering
input_data['date'] = pd.to_datetime(input_data['date'])
input_data['day'] = input_data['date'].dt.day
input_data['month'] = input_data['date'].dt.month

# One-hot encoding
input_data = pd.get_dummies(input_data, columns=['category'])

# Add missing columns
model_columns = ['day', 'month', 'category_Bills',
                 'category_Food', 'category_Shopping', 'category_Travel']

for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[model_columns]

# Prediction
if st.button("Predict Expense"):
    prediction = model.predict(input_data)
    st.success(f"💸 Predicted Expense: ₹{round(prediction[0], 2)}")

# Simple recommendation
st.subheader("📊 Smart Advice")

if st.button("Get Saving Tips"):
    st.write("👉 Reduce spending on non-essential categories like Shopping.")
    st.write("👉 Set monthly budget limits.")
    st.write("👉 Track daily expenses regularly.")
