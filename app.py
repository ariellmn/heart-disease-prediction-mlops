import streamlit as st
import joblib
import pandas as pd

# ===========================
# Load Model
# ===========================
model = joblib.load("model.joblib")

st.title("Heart Disease Prediction App ❤️‍🩹")
st.write("Aplikasi ini memprediksi apakah seseorang berisiko mengidap penyakit jantung.")

# ===========================
# Input Form
# ===========================
st.subheader("Masukkan Data Pasien:")

Age = st.number_input("Age", min_value=1, max_value=120, value=40)
Sex = st.selectbox("Sex", ["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
RestingBP = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=120)
Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
FastingBS = st.selectbox("FastingBS ( > 120 mg/dl )", [0, 1])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
MaxHR = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
ExerciseAngina = st.selectbox("Exercise Angina", ["Y", "N"])
Oldpeak = st.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ===========================
# Prediction
# ===========================
if st.button("Prediksi"):
    input_data = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Pasien kemungkinan memiliki penyakit jantung! (Probabilitas {prob:.2f})")
    else:
        st.success(f"✅ Pasien kemungkinan **TIDAK** memiliki penyakit jantung. (Probabilitas {prob:.2f})")
