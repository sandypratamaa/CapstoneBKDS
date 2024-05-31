import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image

# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
hasil_prediksi = '(none)'
gambar_prediksi = '(none)'

# Load model
model_path = "pplant_leaf_model.h5"
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at {model_path}")

# Define classes
corndiseases_classes = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy", "Corn Northern Leaf Blight", "Non Predict"]

# Set Streamlit configuration
st.set_page_config(page_title="Corn Disease Detection", page_icon=":corn:", layout="wide")

# Sidebar
st.sidebar.title("Corn Disease Detection")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Main content
st.title("Welcome to Corn Disease Detection :corn:")
st.markdown("*Aplikasi ini dapat membantu dalam mengklasifikasi kondisi tanaman jagung anda*")

#add image
IMG_SIZE = (299, 299)
st.image(image="ss.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.markdown("Aplikasi ini berguna untuk mendeteksi penyakit pada tanaman jagung menggunakan teknologi kecerdasan buatan (AI) ...")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    test_image = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.expand_dims(test_image, 0)

    try:
        predictions = model.predict(img_array)
        hasil_prediksi = corndiseases_classes[np.argmax(predictions[0])]
        st.success(f"Prediction: {hasil_prediksi}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.subheader("Penjelasan mengenai jenis-jenis penyakit pada tanaman jagung")
st.markdown("1. Corn Common Rust ...")
st.markdown("2. Corn Gray Leaf Spot ...")
st.markdown("3. Corn Northern Leaf Blight ...")
st.markdown("4. Corn Healthy ...")
