import streamlit as st
import numpy as np
import pickle

# Load model dan encoder
data = pickle.load(open("heart_model.pkl", "rb"))
model = data["model"]
le_sex = data["le_sex"]
le_cp = data["le_cp"]

st.title("Heart Disease Prediction ❤️‍🩹")
st.write("Aplikasi sederhana untuk memprediksi risiko penyakit jantung.")

st.subheader("Masukkan Data Pasien")

# Input user
age = st.number_input("Umur Pasien", min_value=1, max_value=120, value=40)

sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
sex_map = {"Laki-laki": "M", "Perempuan": "F"}
sex_encoded = le_sex.transform([sex_map[sex]])[0]

cp = st.selectbox(
    "Jenis Nyeri Dada",
    [
        "Typical Angina (nyeri khas karena penyempitan arteri)",
        "Atypical Angina (nyeri dada tetapi tidak khas)",
        "Non-Anginal Pain (nyeri bukan dari jantung)",
        "Asymptomatic (tanpa gejala, paling berbahaya)"
    ]
)

cp_map = {
    "Typical Angina (nyeri khas karena penyempitan arteri)": "TA",
    "Atypical Angina (nyeri dada tetapi tidak khas)": "ATA",
    "Non-Anginal Pain (nyeri bukan dari jantung)": "NAP",
    "Asymptomatic (tanpa gejala, paling berbahaya)": "ASY"
}

cp_encoded = le_cp.transform([cp_map[cp]])[0]

chol = st.number_input("Kadar Kolesterol (mg/dl)", min_value=50, max_value=600, value=200)

maxhr = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=220, value=150)

# Prediksi
if st.button("Prediksi Risiko"):
    input_data = np.array([[age, sex_encoded, cp_encoded, chol, maxhr]])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ Kemungkinan **TINGGI** penyakit jantung. (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Kemungkinan **RENDAH** penyakit jantung. (Probabilitas: {prob:.2f})")
