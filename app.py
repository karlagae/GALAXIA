import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# ==========================
# CONFIGURACIÓN
# ==========================
st.set_page_config(page_title="Clasificador de Galaxias", page_icon="✨")

st.title("✨ Clasificación automática de galaxias")
st.write("Modelo CNN entrenado para clasificar galaxias **Elípticas** y **Espirales**.")

# ==========================
# DESCARGA DEL MODELO DESDE DRIVE
# ==========================
MODEL_PATH = "mejor_cnn_galaxias.h5"

# 👉 Pega aquí TU ID real de Google Drive
DRIVE_FILE_ID = "1AbCDeFGHiJKLmnopQRsTuVWxyz"

def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        st.write("Descargando modelo desde Google Drive… (solo la primera vez)")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def cargar_modelo():
    ensure_model_file()
    return tf.keras.models.load_model(MODEL_PATH)

model = cargar_modelo()

IMG_SIZE = (128, 128)
CLASES = {0: "Elíptica", 1: "Espiral"}

# ==========================
# FUNCIÓN DE PREDICCIÓN
# ==========================
def predecir_imagen(img_pil):
    img = img_pil.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    clase = 1 if prob >= 0.5 else 0
    return clase, prob

# ==========================
# INTERFAZ
# ==========================
st.markdown("### 📤 Sube una imagen de una galaxia")

archivo = st.file_uploader("Selecciona una imagen JPG o PNG", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    if st.button("🔮 Clasificar galaxia"):
        clase, prob = predecir_imagen(imagen)

        st.subheader("Resultado")
        st.write(f"**Clase predicha:** {CLASES[clase]}")
        st.write(f"**Probabilidad de espiral:** {prob:.3f}")
        st.write(f"**Probabilidad de elíptica:** {1 - prob:.3f}")

        st.progress(prob)
else:
    st.warning("Sube una imagen para comenzar 🚀")
