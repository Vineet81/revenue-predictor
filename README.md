# 💰 Revenue Predictor App

A web-based app to forecast business revenue based on budget allocations across marketing channels using Ridge and Lasso regression.

## 📌 Problem Statement
Businesses often struggle to decide how much to spend on Email, SEO, Social Media, or Ads. This app predicts expected revenue based on budget allocation, helping businesses optimize their marketing spend.

## 🔍 Features
- Built with Streamlit
- Ridge and Lasso Regression for better generalization
- Interactive sliders to simulate marketing budget
- Feature importance visualization
- Ready to deploy on Streamlit Cloud or Hugging Face Spaces

## 📈 Tech Stack
- Python, Pandas, Scikit-learn, Streamlit, Joblib
- Jupyter for EDA and model training

## 📊 Dataset (Synthetic)
Synthetic dataset created for educational purposes with variables:
- `Email_Spend`, `Social_Spend`, `SEO_Spend`, `PaidAds_Spend`, and `Revenue`.

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

