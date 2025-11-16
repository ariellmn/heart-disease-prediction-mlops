import streamlit as st
import joblib
import pandas as pd

# LOoad Model
model = joblib.load("model.joblib")

st.title("Heart Disease Prediction ❤️")
st.write("Masukkan beberapa data sederhana untuk memprediksi risiko penyakit jantung.")

st.subheader("Input Data Pasien")

# Input User
age = st.number_input(
    "Umur Pasien",
    min_value=1,
    max_value=120,
    value=40,
    help="Semakin muda biasanya semakin rendah risikonya (<40 = rendah, 40–60 = sedang, >60 = tinggi)"
)

sex = st.selectbox(
    "Jenis Kelamin",
    ["Laki-laki", "Perempuan"],
    help="Pada dataset, laki-laki cenderung memiliki risiko lebih tinggi."
)

sex_map = {"Laki-laki": "M", "Perempuan": "F"}

cp = st.selectbox(
    "Jenis Nyeri Dada",
    [
        "Typical Angina (nyeri khas karena penyempitan arteri)",
        "Atypical Angina (nyeri dada tetapi tidak khas)",
        "Non-Anginal Pain (nyeri bukan dari jantung)",
        "Asymptomatic (tanpa gejala)"
    ]
)

cp_map = {
    "Typical Angina (nyeri khas karena penyempitan arteri)": "TA",
    "Atypical Angina (nyeri dada tetapi tidak khas)": "ATA",
    "Non-Anginal Pain (nyeri bukan dari jantung)": "NAP",
    "Asymptomatic (tanpa gejala)": "ASY"
}

chol = st.number_input(
    "Kadar Kolesterol (mg/dl)",
    min_value=80,
    max_value=600,
    value=180,
    help="• <200 = rendah\n• 200–240 = sedang\n• >240 = tinggi"
)

maxhr = st.number_input(
    "Detak Jantung Maksimum (MaxHR)",
    min_value=60,
    max_value=220,
    value=160,
    help="Semakin tinggi MaxHR → semakin rendah risiko (≥150 bagus di dataset)"
)

# Prediction
if st.button("Prediksi Risiko"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex_map[sex],
        "ChestPainType": cp_map[cp],
        "RestingBP": 120,          
        "Cholesterol": chol,
        "FastingBS": 0,            
        "RestingECG": "Normal",    
        "MaxHR": maxhr,
        "ExerciseAngina": "N",     
        "Oldpeak": 1.0,           
        "ST_Slope": "Flat"         
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Risiko TINGGI terkena penyakit jantung. (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Risiko RENDAH terkena penyakit jantung. (Probabilitas: {prob:.2f})")

