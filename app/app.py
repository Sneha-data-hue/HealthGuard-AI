import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('../models/rf_model.pkl', 'rb'))

# Sidebar
st.sidebar.title(" HealthGuard AI")
st.sidebar.write("This app predicts heart disease risk using ML models.")

# Title
st.title(" Heart Disease Prediction App")
st.markdown("###  AI-powered system to predict heart disease risk")
st.write("Fill in patient details and click Predict")

st.write("### Enter patient details:")

# Layout (Columns)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 50)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)

with col2:
    thalch = st.slider("Max Heart Rate", 60, 200, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

# Other inputs
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal"])
fbs = st.selectbox("Fasting Blood Sugar", [True, False])
exang = st.selectbox("Exercise Angina", [True, False])
restecg = st.selectbox("Rest ECG", ["normal", "st-t abnormality"])
slope = st.selectbox("Slope", ["flat", "upsloping"])

# Convert inputs to dataframe
input_data = pd.DataFrame({
    'age': [age],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'thalch': [thalch],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'sex_Male': [1 if sex == "Male" else 0],
    'cp_atypical angina': [1 if cp == "atypical angina" else 0],
    'cp_non-anginal': [1 if cp == "non-anginal" else 0],
    'cp_typical angina': [1 if cp == "typical angina" else 0],
    'restecg_normal': [1 if restecg == "normal" else 0],
    'restecg_st-t abnormality': [1 if restecg == "st-t abnormality" else 0],
    'slope_flat': [1 if slope == "flat" else 0],
    'slope_upsloping': [1 if slope == "upsloping" else 0]
})

# Title above button
st.markdown("###  Click to Predict")

# Predict button
if st.button("Predict Risk"):
    
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result_text = "High Risk of Heart Disease"
        st.error(" High Risk of Heart Disease")
        st.progress(85)
    else:
        result_text = "Low Risk of Heart Disease"
        st.success(" Low Risk of Heart Disease")
        st.progress(30)

    # Create result dataframe
    result_df = input_data.copy()
    result_df['Prediction'] = result_text

    # Show table
    st.write("###  Result Summary")
    st.dataframe(result_df)

    # Download button
    csv = result_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Result as CSV",
        data=csv,
        file_name='heart_disease_result.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown("Developed by Sneha Sunny | AI Project for Healthcare ")