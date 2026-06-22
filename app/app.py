import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Prediction System")
st.markdown("Enter the patient details below to predict heart disease risk.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"
    ])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", [
        "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"
    ])

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", [
        "Upsloping", "Flat", "Downsloping"
    ])
    ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", [
        "Normal", "Fixed Defect", "Reversible Defect"
    ])

# Convert inputs
sex = 1 if sex == "Male" else 0
cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
fbs = 1 if fbs == "Yes" else 0
restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
exang = 1 if exang == "Yes" else 0
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

# Predict button
if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease — Confidence: {probability[1]*100:.1f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease — Confidence: {probability[0]*100:.1f}%")

    st.markdown("### Prediction Confidence")
    st.progress(float(probability[1]))
    st.caption(f"Disease probability: {probability[1]*100:.1f}%")