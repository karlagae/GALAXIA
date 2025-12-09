import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# ==========================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================
st.set_page_config(
    page_title="Clasificador de Galaxias",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Clasificación automática de galaxias")
st.markdown(
    """
    Esta aplicación utiliza una **Red Neuronal Convolucional (CNN)** entrenada para
    clasificar imágenes de galaxias en dos categorías principales:

    - 🌌 **Galaxias Elípticas**  
    - 🌀 **Galaxias Espirales**

    Sube una imagen de una galaxia y el modelo estimará la probabilidad de que
    pertenezca a cada clase.
    """
)

# ==========================
# INFORMACIÓN EN LA SIDEBAR
# ==========================
st.sidebar.title("ℹ️ Acerca del modelo")
st.sidebar.markdown(
    """
    **Modelo:** CNN sencilla con 2 bloques Conv2D + MaxPooling  
    **Tamaño de entrada:** 128 × 128 × 3 (RGB)  
    **Tarea:** Clasificación binaria (Elíptica / Espiral)  

    **Desempeño (ejemplo):**
    - Exactitud en validación: ~95 %
    - Overfitting controlado con:
        - Dropout
        - EarlyStopping
        - ReduceLROnPlateau
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado como proyecto de **clasificación automática de galaxias** ⭐")

# ==========================
# DESCARGA Y CARGA DEL MODELO
# ==========================
MODEL_PATH = "mejor_cnn_galaxias.h5"
DRIVE_FILE_ID = "1dPFzrqdKQZzqtO_IBFaLNWH9hDud6Z8z"  # tu ID

def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Descargando modelo desde Google Drive (solo la primera vez)…")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        try:
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error("❌ No se pudo descargar el modelo desde Drive.")
            st.error("Verifica que el archivo sea público y vuelve a intentar.")
            st.stop()

@st.cache_resource
def cargar_modelo():
    ensure_model_file()
    return tf.keras.models.load_model(MODEL_PATH)

with st.spinner("Cargando modelo CNN…"):
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
    prob = float(model.predict(arr, verbose=0)[0][0])  # prob de ESPIRAL
    clase = 1 if prob >= 0.5 else 0
    return clase, prob

# ==========================
# INTERFAZ PRINCIPAL
# ==========================
st.markdown("### 📤 Sube una imagen de una galaxia")

col1, col2 = st.columns([1.2, 1])

archivo = st.file_uploader(
    "Selecciona una imagen JPG o PNG",
    type=["jpg", "jpeg", "png"]
)

with col1:
    if archivo is not None:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen cargada", use_column_width=True)
    else:
        st.info("Sube una imagen para comenzar 🚀")

with col2:
    if archivo is not None:
        if st.button("🔮 Clasificar galaxia", use_container_width=True):
            clase, prob_spiral = predecir_imagen(imagen)
            prob_elliptical = 1 - prob_spiral

            # Texto principal de resultado
            st.subheader("Resultado de la clasificación")

            icono = "🌀" if clase == 1 else "🌌"
            nombre_clase = CLASES[clase]

            st.markdown(
                f"""
                ### {icono} Galaxia **{nombre_clase}**

                - Probabilidad de **Espiral**: `{prob_spiral:.3f}`
                - Probabilidad de **Elíptica**: `{prob_elliptical:.3f}`
                """
            )

            # Barra de probabilidad (convertimos a 0–100)
            st.markdown("**Confianza en clase Espiral:**")
            st.progress(int(prob_spiral * 100))

            st.caption(
                "Las probabilidades se estiman a partir de la salida sigmoide del modelo."
            )
    else:
        # Si no hay archivo, no mostramos botón aquí
        pass
