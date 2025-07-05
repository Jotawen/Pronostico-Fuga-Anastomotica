
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelo y columnas
model = joblib.load("modelo_random_forest.pkl")
columnas = joblib.load("columnas_modelo.pkl")

st.title("Predicción de Fuga Anastomótica")
st.markdown("Ingrese los datos del paciente para estimar el riesgo.")

# Variables esenciales (personalizables)
edad = st.number_input("Edad", min_value=18, max_value=100, value=60)
imc = st.number_input("IMC", min_value=10.0, max_value=60.0, value=22.0)
tiempo_qx = st.number_input("Tiempo quirúrgico (min)", min_value=0, max_value=600, value=180)
sangrado = st.number_input("Sangrado estimado (ml)", min_value=0, max_value=5000, value=200)

# Categóricas más comunes
sexo = st.selectbox("Sexo", ["F", "M"])
reconstruccion = st.selectbox("Tipo de reconstrucción", ["DESCONOCIDO", "BILLROTH II", "ROUX EN Y", "BILLROTH I"])
abordaje = st.selectbox("Abordaje quirúrgico", ["DESCONOCIDO", "ABIERTA", "LAPAROSCOPICA"])

# Crear vector de entrada
input_dict = {
    "EDAD": edad,
    "IMC": imc,
    "TIEMPO_QX": tiempo_qx,
    "SANGRADO": sangrado,
    "SEXO_" + sexo: 1,
    "RECONSTRUCCIÓN_" + reconstruccion.upper(): 1,
    "ABORDAJE_\nABIERTA/_LA_" + abordaje.upper(): 1,
}

# Generar vector de entrada con ceros y llenar lo requerido
X_input = pd.DataFrame([np.zeros(len(columnas))], columns=columnas)
for key, value in input_dict.items():
    if key in X_input.columns:
        X_input[key] = value

# Predicción
if st.button("Calcular riesgo"):
    proba = model.predict_proba(X_input)[0][1]
    st.success(f"Probabilidad estimada de fuga anastomótica: {proba*100:.2f}%")
