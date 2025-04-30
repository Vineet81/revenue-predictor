# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('revenue_model.pkl')

# Title
st.title("💰 Revenue Predictor")

# User Inputs
st.subheader("Enter your marketing spend data:")
email = st.number_input("📧 Email Spend ($)", min_value=0.0, format="%.2f")
social = st.number_input("📱 Social Media Spend ($)", min_value=0.0, format="%.2f")
seo = st.number_input("🔍 SEO Spend ($)", min_value=0.0, format="%.2f")
ads = st.number_input("📢 Paid Ads Spend ($)", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict Revenue 💸"):
    input_data = pd.DataFrame([[email, social, seo, ads]],
        columns=["Email_Spend", "Social_Spend", "SEO_Spend", "PaidAds_Spend"])
    
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Revenue: ${prediction:,.2f}")

