# Streamlit App for Customer Churn Prediction

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("xgb_churn_model_best.pkl")
scaler = joblib.load("scaler.pkl")  # Make sure you saved your scaler too

st.title("🔮 Customer Churn Prediction App")

st.write("Fill in customer details to predict whether they are likely to churn.")

# Input sliders
premium = st.slider("Premium Amount (SAR)", 500, 50000, 10000)
account_length = st.slider("Account Length (Months)", 1, 120, 24)
number_of_transactions = st.slider("Number of Transactions", 0, 100, 10)
transaction_amount = st.slider("Total Transaction Amount", 0, 100000, 20000)
payment_delay_instances = st.slider("Payment Delay Instances", 0, 10, 1)
promotional_offers_accepted = st.slider("Promotional Offers Accepted", 0, 5, 1)
online_banking_usage = st.slider("Online Banking Usage", 0, 30, 10)
mobile_app_usage = st.slider("Mobile App Usage", 0, 30, 10)
customer_support_calls = st.slider("Customer Support Calls", 0, 10, 1)
complaints_filed = st.slider("Complaints Filed", 0, 5, 0)
avg_transaction_amount = st.slider("Average Transaction Amount", 0, 50000, 2000)
delay_ratio = st.slider("Delay Ratio", 0, 100000, 10000)

# Create dataframe
input_df = pd.DataFrame({
    "premium": [premium],
    "account_length": [account_length],
    "number_of_transactions": [number_of_transactions],
    "transaction_amount": [transaction_amount],
    "payment_delay_instances": [payment_delay_instances],
    "promotional_offers_accepted": [promotional_offers_accepted],
    "online_banking_usage": [online_banking_usage],
    "mobile_app_usage": [mobile_app_usage],
    "customer_support_calls": [customer_support_calls],
    "complaints_filed": [complaints_filed],
    "avg_transaction_amount": [avg_transaction_amount],
    "delay_ratio": [delay_ratio]
})

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")
