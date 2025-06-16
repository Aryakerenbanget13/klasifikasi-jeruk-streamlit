import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# Unduh model dari Google Drive jika belum ada
model_path = 'model_kematangan.h5'
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id=1A2B3cXYZ456abc'  # Ganti dengan ID milikmu
    gdown.download(url, model_path, quiet=False)

# Load model CNN
model = tf.keras.models.load_model(model_path)

# Label kelas
class_labels = ['Jeruk Belum Matang', 'Jeruk Busuk', 'Jeruk Matang', 'Non Jeruk']

# Fungsi preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Tampilan GUI
st.set_page_config(page_title="Klasifikasi Jeruk", layout="centered")
st.title("🍊 Klasifikasi Tingkat Kematangan Jeruk")
st.write("Upload gambar jeruk, dan sistem akan memprediksi tingkat kematangannya.")

uploaded_file = st.file_uploader("Upload gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    st.write("Memproses gambar dan melakukan prediksi...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    
    pred_index = np.argmax(prediction)
    pred_label = class_labels[pred_index]
    confidence = prediction[pred_index] * 100

    st.success(f"Prediksi: **{pred_label}** ({confidence:.2f}%)")
