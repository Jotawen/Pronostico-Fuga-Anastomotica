import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y columnas
model = joblib.load("modelo_random_forest.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

# Título y descripción
st.title("📊 Predicción de Fuga Anastomótica")
st.write(
    """
    Esta aplicación permite visualizar los resultados del modelo de predicción de fuga anastomótica 
    utilizando datos previamente cargados.
    """
)

# Cargar automáticamente el archivo Excel desde el entorno local
try:
    df = pd.read_excel("Gastrectomias.xlsx", sheet_name="TOTAL GASTRECTOMIAS")
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
except FileNotFoundError:
    st.error("❌ No se encuentra el archivo Gastrectomias.xlsx en el directorio.")
    st.stop()

# Procesamiento
numeric_columns = ["EDAD", "IMC", "TIEMPO_QX", "SANGRADO", "ESTANCIA_HOSPITALARIA_(DÍAS)", "DIAS_EN_UCI"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").str.extract("([0-9.]+)").astype(float)

df["FUGA_ANASTOMOTICA"] = df["FUGA_ANASTOMOTICA"].str.upper().map({"SI": 1, "NO": 0})
df_model = df.dropna(subset=["FUGA_ANASTOMOTICA"])

exclude_cols = ["NOMBRE", "CÉDULA", "FECHA_CX", "FUGA_ANASTOMOTICA", "TIPO_DE_FISTULA", "CAUSA_REINGRESO", "CONSIDERACIONES"]
features = [col for col in df_model.columns if col not in exclude_cols and col in columnas_modelo]

X = df_model[features].copy()
y = df_model["FUGA_ANASTOMOTICA"]

# Completar valores faltantes
num_cols = X.select_dtypes(include=["float64", "int64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("DESCONOCIDO")

# Codificación dummy
X_encoded = pd.get_dummies(X)
missing_cols = [col for col in columnas_modelo if col not in X_encoded.columns]
for col in missing_cols:
    X_encoded[col] = 0
X_encoded = X_encoded[columnas_modelo]

# Predicción
predicciones = model.predict(X_encoded)
df_model["PREDICCION_FUGA"] = predicciones

# Resultados
conteo = df_model["PREDICCION_FUGA"].value_counts().sort_index()
st.success(f"✔️ Pacientes sin fuga: {conteo.get(0, 0)}")
st.error(f"❗ Pacientes con fuga: {conteo.get(1, 0)}")

# Mostrar tabla
st.subheader("🔍 Resultados detallados")
st.dataframe(df_model[["NOMBRE", "CÉDULA", "PREDICCION_FUGA"] + features])

# Opción de descarga
csv = df_model.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar resultados como CSV",
    data=csv,
    file_name="resultados_fuga.csv",
    mime="text/csv",
)
