import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Label Encoders (harus sama urutan dan label seperti saat training)
gender_encoder = LabelEncoder()
gender_encoder.fit(['Male', 'Female'])

education_encoder = LabelEncoder()
education_encoder.fit(['High School', 'College', 'Graduate', 'Other'])

home_encoder = LabelEncoder()
home_encoder.fit(['Rent', 'Own', 'Mortgage', 'Other'])

intent_encoder = LabelEncoder()
intent_encoder.fit(['Personal', 'Medical', 'Education', 'Venture', 'Home Improvement', 'Debt Consolidation'])

# Judul
st.title("Prediksi Persetujuan Pinjaman")

st.markdown("Masukkan detail pemohon untuk memprediksi apakah pinjaman akan **disetujui** atau **ditolak**.")

# Form Input
with st.form("form_prediksi"):
    person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
    person_gender = st.selectbox("Jenis Kelamin", gender_encoder.classes_)
    person_education = st.selectbox("Pendidikan Terakhir", education_encoder.classes_)
    person_income = st.number_input("Pendapatan Tahunan", min_value=1000, value=45000)
    person_emp_exp = st.number_input("Pengalaman Kerja (tahun)", min_value=0, value=5)
    person_home_ownership = st.selectbox("Status Tempat Tinggal", home_encoder.classes_)
    loan_amnt = st.number_input("Jumlah Pinjaman", min_value=500, value=10000)
    loan_intent = st.selectbox("Tujuan Pinjaman", intent_encoder.classes_)
    loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", min_value=1.0, value=11.5)
    loan_percent_income = st.number_input("Rasio Pinjaman terhadap Pendapatan", min_value=0.01, max_value=1.0, value=0.22)
    cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", min_value=1, value=6)
    credit_score = st.number_input("Skor Kredit", min_value=300, max_value=850, value=730)
    previous_loan_defaults_on_file = st.selectbox("Pernah Menunggak Pinjaman Sebelumnya?", [0, 1])

    submitted = st.form_submit_button("Prediksi")

    if submitted:
        df_pred = pd.DataFrame([{
            'person_age': person_age,
            'person_gender': gender_encoder.transform([person_gender])[0],
            'person_education': education_encoder.transform([person_education])[0],
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_home_ownership': home_encoder.transform([person_home_ownership])[0],
            'loan_amnt': loan_amnt,
            'loan_intent': intent_encoder.transform([loan_intent])[0],
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }])

        pred = model.predict(df_pred)[0]
        hasil = " Disetujui" if pred == 1 else " Ditolak"
        st.subheader(f"Hasil Prediksi: {hasil}")

st.markdown("### Contoh Test Case")
if st.button("Test Case 1"):
    st.info("Pria, umur 30, pendapatan 45K, tujuan pribadi, skor kredit 730")
    st.success("Prediksi:  Disetujui (kemungkinan besar)")

if st.button("Test Case 2"):
    st.info("Wanita, umur 22, pendapatan 12K, tunggakan sebelumnya, skor kredit 580")
    st.error("Prediksi:  Ditolak (kemungkinan besar)")
