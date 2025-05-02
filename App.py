import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Función para predecir el dígito
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# Configuración de la página
st.set_page_config(page_title='🔢 Reconocimiento de Dígitos a Mano', layout='wide')
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🔢 Reconocimiento de Dígitos Escritos a Mano</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>🖌️ Dibuja un número del 0 al 9 en el panel y presiona <b>'Predecir'</b></h3>", unsafe_allow_html=True)
st.markdown("---")

# Opciones de dibujo
st.sidebar.header("🛠️ Ajustes del dibujo")
stroke_width = st.sidebar.slider('🖊️ Ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Botón de predicción
st.markdown("### 👉 Cuando estés listo, presiona el botón para predecir:")
if st.button('🚀 Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.success(f"✅ ¡El dígito reconocido es: **{res}**!")
    else:
        st.warning("⚠️ Por favor, dibuja un dígito antes de predecir.")

# Información lateral
st.sidebar.markdown("---")
st.sidebar.header("📌 Acerca de")
st.sidebar.markdown("""
Esta aplicación utiliza una **Red Neuronal Artificial (RNA)** entrenada para reconocer dígitos escritos a mano 🧠✍️.

Creado con 💻 usando Streamlit y TensorFlow.

Basado en el desarrollo de **Vinay Uniyal**.
""")
# st.sidebar.markdown("[Repositorio en GitHub](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")
