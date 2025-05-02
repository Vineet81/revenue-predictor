#app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("revenue_model.pkl")

st.title("Marketing Spend – Revenue Prediction Dashboard")

# Create tabs for better UI
tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])

# -------- Tab 1: Single Input Prediction --------
with tab1:
    st.header("Predict Revenue from Your Inputs")
    st.markdown("Use sliders to set marketing spend:")
    
    email = st.slider("Email Spend ($)", 0, 15000, 3000)
    social = st.slider("Social Media Spend ($)", 0, 10000, 1000)
    seo = st.slider("SEO Spend ($)", 0, 10000, 1000)
    ads = st.slider("Paid Ads Spend ($)", 0, 20000, 5000)
    
    input_data = pd.DataFrame({
        'Email_Spend': [email],
        'Social_Spend': [social],
        'SEO_Spend': [seo],
        'PaidAds_Spend': [ads]
       
    })

    st.subheader("Input Summary")
    st.dataframe(input_data)

    prediction = model.predict(input_data)[0]
    st.metric(label="Predicted Revenue ($)", value=f"{prediction:,.2f}")

    st.subheader("Spend Distribution")
    fig, ax = plt.subplots()
    labels = ['Email', 'Social Media', 'SEO', 'Paid Ads']
    values = [email,social, seo, ads]
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)

# -------- Tab 2: Bulk CSV Upload --------
with tab2:
    st.header("Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(user_data)

        try:
            predictions = model.predict(user_data)
            user_data['Predicted Revenue'] = predictions
            st.subheader("Predicted Results")
            st.dataframe(user_data)

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(user_data)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='revenue_predictions.csv',
                mime='text/csv',
            )

            # Visualization
            st.subheader("Revenue vs. Paid Ads")
            fig2, ax2 = plt.subplots()
            ax2.scatter(user_data['PaidAds_Spend'], user_data['Predicted Revenue'], color='green')
            ax2.set_xlabel("Paid Ads")
            ax2.set_ylabel("Predicted Revenue")
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
