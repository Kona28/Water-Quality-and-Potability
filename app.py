import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# Load model dan scaler yang sudah disimpan
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💧 Water Quality Potability Prediction & Monitoring")
st.write("Aplikasi Machine Learning untuk memprediksi kelayakan air minum berdasarkan parameter kualitas air.")

# Sidebar untuk input parameter
st.sidebar.header("Masukkan Parameter Air:")
ph = st.sidebar.number_input("pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.number_input("Hardness (Kekerasan)", 0.0, 350.0, 200.0)
solids = st.sidebar.number_input("Solids (Total Padatan Terlarut)", 0.0, 70000.0, 20000.0)
chloramines = st.sidebar.number_input("Chloramines", 0.0, 15.0, 7.0)
sulfate = st.sidebar.number_input("Sulfate", 0.0, 500.0, 330.0)
conductivity = st.sidebar.number_input("Conductivity", 0.0, 800.0, 420.0)
organic_carbon = st.sidebar.number_input("Organic Carbon", 0.0, 30.0, 14.0)
trihalomethanes = st.sidebar.number_input("Trihalomethanes", 0.0, 130.0, 66.0)
turbidity = st.sidebar.number_input("Turbidity (Kekeruhan)", 0.0, 8.0, 4.0)

if st.button("Prediksi Kualitas Air"):
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]],
                              columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
    
    # Standarisasi data input
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    result = "Layak Diminum (Potable)" if prediction[0] == 1 else "Tidak Layak Diminum (Not Potable)"
    
    if prediction[0] == 1:
        st.success(f"Hasil Prediksi: **{result}**")
    else:
        st.error(f"Hasil Prediksi: **{result}**")
    
    st.write(f"Tingkat Keyakinan (Probability): {np.max(probability)*100:.2f}%")
    
    # Logging sederhana untuk Monitoring
    log_data = input_data.copy()
    log_data['Prediction'] = result
    log_data['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_file = 'prediction_log.csv'
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, log_data], ignore_index=True)
    else:
        log_df = log_data
    log_df.to_csv(log_file, index=False)

# Dashboard Monitoring Sederhana
st.subheader("📊 Sistem Monitoring & Riwayat Prediksi")
if os.path.exists('prediction_log.csv'):
    log_df = pd.read_csv('prediction_log.csv')
    st.metric(label="Total Prediksi Dilakukan", value=len(log_df))
    st.write("Riwayat Prediksi Terakhir:")
    st.dataframe(log_df.tail(10))
else:
    st.info("Belum ada data monitoring yang tercatat.")