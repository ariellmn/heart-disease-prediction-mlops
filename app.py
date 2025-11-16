import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("model.joblib")

st.title("Heart Disease Prediction ❤️")
st.write("Masukkan data pasien, lihat prediksi risiko, dan cek rekomendasi rentang nilai yang aman.")

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
        "Typical Angina — nyeri dada khas akibat penyempitan arteri (risiko tinggi)",
        "Atypical Angina — nyeri dada tidak khas (risiko sedang)",
        "Non-Anginal Pain — nyeri bukan dari jantung (risiko rendah)",
        "Asymptomatic — tidak ada gejala (risiko paling rendah)"
    ]
)

cp_map = {
    "Typical Angina — nyeri dada khas akibat penyempitan arteri (risiko tinggi)": "TA",
    "Atypical Angina — nyeri dada tidak khas (risiko sedang)": "ATA",
    "Non-Anginal Pain — nyeri bukan dari jantung (risiko rendah)": "NAP",
    "Asymptomatic — tidak ada gejala (risiko paling rendah)": "ASY"
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


st.write("---")
st.subheader("Rekomendasi Rentang Aman (berdasarkan dataset klinis)")
st.write(
"""
**Umur:**  
- Rendah: <40  
- Sedang: 40–60  
- Tinggi: >60  

**Kolesterol:**  
- Baik: 100–200  
- Sedang: 200–240  
- Bahaya: >240  

**MaxHR:**  
- Baik: 150–200  
- Sedang: 120–150  
- Rendah: <120  
"""
)
st.write("---")


# Prediksi
if st.button("Prediksi Risiko ❤️"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex_map[sex],
        "ChestPainType": cp_map[cp],
        "RestingBP": 120,          # Nilai konstan aman
        "Cholesterol": chol,
        "FastingBS": 0,            # Normal
        "RestingECG": "Normal",    
        "MaxHR": maxhr,
        "ExerciseAngina": "N",     # Tidak ada angina saat olahraga
        "Oldpeak": 0.0,            # Tidak ada ST depresion
        "ST_Slope": "Up"           # Up-sloping: risiko paling rendah
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(
            f"⚠️ Risiko TINGGI terkena penyakit jantung.\n\n"
            f"Probabilitas: **{prob:.2f}**"
        )
    else:
        st.success(
            f"✅ Risiko RENDAH terkena penyakit jantung.\n\n"
            f"Probabilitas: **{prob:.2f}**"
        )

