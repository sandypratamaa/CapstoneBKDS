import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import base64

# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
hasil_prediksi = '(none)'
gambar_prediksi = '(none)'

# Load model
model = tf.keras.models.load_model("modelcorn.h5")

# Define classes
corndiseases_classes = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy", "Corn Northern Leaf Blight"]

# Set Streamlit configuration
st.set_page_config(page_title="Corn Disease Detection", page_icon=":corn:", layout="wide")

# Sidebar
st.sidebar.title("Corn Disease Detection")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Main content
st.title("Welcome to Corn Disease Detection :corn:")
st.caption("**Aplikasi ini dapat membantu dalam mengklasifikasi kondisi tanaman jagung anda**")

#add image
# Define image size
IMG_SIZE = (299, 299)
st.image(image="ss.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

#penjelasan
st.caption(":black[**Terdapat 4 jenis kategori yang aplikasi dapat deteksi yaitu Corn Common Rust, Corn Northern, Corn Gray Leaf Spot, dan Corn Healty yang akan di proses dibawah ini**]")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict
  
    test_image = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.expand_dims(test_image, 0)

    predictions = model.predict(img_array)
    hasil_prediksi = corndiseases_classes[np.argmax(predictions[0])]

    # Display result
    st.success(f"Prediction: {hasil_prediksi}")

st.subheader(":black[**Penjelasan mengenai jenis-jenis penyakit pada tanaman jagung**]")

st.caption(":black[**1.Corn Common Rust adalah penyakit tanaman jagung yang disebabkan oleh jamur bernama Puccinia sorghi. Penyakit ini sangat umum terjadi pada jagung dan dapat menyebabkan kerusakan pada tanaman, yang pada akhirnya dapat menyebabkan penurunan hasil panen.**]")
st.caption(":black[**2.Corn Gray Leaf Spot adalah penyakit tanaman jagung yang disebabkan oleh jamur bernama Cercospora zeae-maydis. Penyakit ini dapat menyebabkan kerusakan pada daun jagung dan dapat mengurangi hasil panen pada saat panen.**]")
st.caption(":black[**3.Corn Northern Leaf Blight adalah penyakit tanaman jagung yang disebabkan oleh jamur bernama Exserohilum turcicum (dahulu disebut sebagai Helminthosporium turcicum). Penyakit ini dapat menyebabkan kerusakan pada daun jagung dan dapat mengurangi hasil panen.**]")
st.caption(":black[**4.Corn Healty adalah kondisi bahwa tanaman jagung anda sehat.**]")
