import streamlit as st
import pandas as pd
import joblib
import io

st.set_page_config(page_title="Pronóstico Fuga Anastomótica", layout="centered")

st.title("🧠 Pronóstico de Fuga Anastomótica")
st.markdown("""
Este modelo predice el riesgo de fuga anastomótica postoperatoria en base a datos clínicos prequirúrgicos y perioperatorios.
Sube un archivo Excel con los datos de los pacientes.
""")

# Cargar modelo y columnas esperadas
modelo = joblib.load("modelo_random_forest.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

# Subir archivo
archivo = st.file_uploader("📂 Sube tu archivo Excel con los datos", type=["xlsx"])

if archivo:
    try:
        df = pd.read_excel(archivo)
        st.success("Archivo cargado exitosamente ✅")
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())

        # Estandarizar nombres de columnas
        df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

        # Validar columnas requeridas
        columnas_faltantes = [col for col in columnas_modelo if col not in df.columns]
        if columnas_faltantes:
            st.error(f"❌ Faltan las siguientes columnas requeridas: {columnas_faltantes}")
        else:
            # Ordenar y seleccionar columnas correctas
            X = df[columnas_modelo]

            # Realizar predicción
            predicciones = modelo.predict(X)
            df_resultado = df.copy()
            df_resultado['PREDICCION_FUGA'] = predicciones

            st.success("Predicciones realizadas exitosamente ✅")
            st.subheader("📊 Resultados")
            st.dataframe(df_resultado[['PREDICCION_FUGA']].value_counts().rename("Pacientes").reset_index())

            st.subheader("📋 Detalle de las predicciones")
            st.dataframe(df_resultado)

            # Botón de descarga
            output = io.BytesIO()
            df_resultado.to_excel(output, index=False)
            st.download_button("📥 Descargar archivo con predicciones", data=output.getvalue(), file_name="predicciones_fuga.xlsx")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
else:
    st.info("Por favor, sube un archivo Excel para comenzar.")
