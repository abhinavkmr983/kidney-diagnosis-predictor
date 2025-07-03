import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler from .joblib files
model = joblib.load("models/kidney_ckd_model.joblib")
scaler = joblib.load("models/kidney_ckd_scaler.joblib")

# ---------------------------
# Streamlit UI Configuration
# ---------------------------
st.set_page_config(page_title="AI Kidney Disease Predictor", page_icon="🌡️", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f8;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #003366;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 18px;
        margin-bottom: 15px;
    }
    .predict-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 0 12px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌡️ AI-Powered Kidney Disease Predictor</div>", unsafe_allow_html=True)
st.markdown("### Enter Patient Health Metrics Below:")

# ---------------------------
# Input Form
# ---------------------------
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 1, 100, 35)
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian", "Other"])

    with col2:
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 5.0, step=0.1, value=1.2)
        eGFR = st.number_input("eGFR (ml/min/1.73m²)", 5.0, 120.0, step=1.0, value=75.0)
        bun = st.number_input("Blood Urea Nitrogen (mg/dL)", 5.0, 40.0, step=0.5, value=16.0)

    with col3:
        urine_cr = st.number_input("Urine Creatinine (mg/dL)", 20.0, 300.0, step=1.0, value=100.0)
        acr = st.number_input("Urine ACR (mg/g)", 5.0, 1000.0, step=1.0, value=30.0)
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])

    sbp = st.slider("Systolic BP (mm Hg)", 90, 200, 130)
    dbp = st.slider("Diastolic BP (mm Hg)", 50, 130, 85)

    submitted = st.form_submit_button("🔎 Predict CKD Risk")

# ---------------------------
# Inference Logic
# ---------------------------
if submitted:
    try:
        # Encode categorical inputs
        sex_enc = 1 if sex == "Male" else 0
        race_enc = {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other": 4}[race]
        diab_enc = 1 if diabetes == "Yes" else 0

        # Prepare and scale input
        input_array = np.array([[age, creatinine, eGFR, urine_cr, acr, bun, sbp, dbp, sex_enc, race_enc, diab_enc]])
        input_scaled = scaler.transform(input_array)

        # Prediction
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        # Result display
        st.markdown("<hr>", unsafe_allow_html=True)
        if pred == 1:
            st.error(f"⚠️ High Risk of Chronic Kidney Disease Detected (Probability: {proba*100:.2f}%)")
            st.markdown("""
            ### 🔍 AI Recommendations:
            - Visit a nephrologist urgently.
            - Monitor blood pressure and reduce salt intake.
            - Review medications for kidney safety.
            - Get further tests: Ultrasound, 24-hour ACR.
            - Maintain hydration and reduce protein overload.
            """)
        else:
            st.success(f"✅ No Chronic Kidney Disease Detected (Probability: {proba*100:.2f}%)")
            st.markdown("""
            ### 💡 Health Suggestions:
            - Maintain current health routine.
            - Annual renal checkups advised.
            - Stay hydrated and avoid nephrotoxic substances.
            - Maintain blood sugar and pressure in range.
            """)
    except Exception as e:
        st.error(f"An error occurred: {e}")
