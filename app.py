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
st.markdown("**Aplikasi ini dapat membantu dalam mengklasifikasi kondisi tanaman jagung anda**")

# Add image
# Define image size
IMG_SIZE = (299, 299)
st.image(image="ss.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Penjelasan
st.markdown("Aplikasi ini berguna untuk mendeteksi penyakit pada tanaman jagung menggunakan teknologi kecerdasan buatan (AI) dan algoritma deep learning untuk mendiagnosis penyakit pada tanaman jagung melalui gambar yang diunggah ke aplikasi ini. Dataset yang digunakan dalam sistem ini terdiri dari ribuan gambar tanaman jagung yang terinfeksi penyakit dan sehat. Saat pengguna mengunggah gambar tanaman jagung, sistem akan menganalisis gambar tersebut dan memberikan diagnosis. Algoritma deep learning digunakan karena dapat mempelajari fitur-fitur kompleks yang terkait dengan penyakit pada tanaman jagung dan menghasilkan diagnosis yang lebih akurat.")

st.markdown(":corn: Terdapat 4 jenis kategori yang aplikasi dapat deteksi yaitu Corn Common Rust, Corn Northern Leaf Blight, Corn Gray Leaf Spot, dan Corn Healty yang akan di proses dibawah ini : ")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict
    test_image = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.array(test_image)

    # Check if the uploaded image is a leaf or not (Improved heuristic check)
    def is_leaf(image_array):
        green_threshold = 100
        green_pixels = np.sum((image_array[:,:,1] > green_threshold) & 
                              (image_array[:,:,0] < green_threshold) & 
                              (image_array[:,:,2] < green_threshold))
        total_pixels = image_array.shape[0] * image_array.shape[1]
        green_proportion = green_pixels / total_pixels
        return green_proportion > 0.1  # Adjust this threshold based on your dataset

    if is_leaf(img_array):
        img_array = np.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        hasil_prediksi = corndiseases_classes[np.argmax(predictions[0])]
        # Display result
        st.success(f"Prediction: {hasil_prediksi}")
    else:
        st.error("Gambar tidak dapat diprediksi. Silakan unggah gambar daun jagung.")

st.subheader(" Penjelasan mengenai jenis-jenis penyakit pada tanaman jagung ")

st.markdown("1. Corn Common Rust atau karat jagung adalah penyakit yang disebabkan oleh jamur Puccinia sorghi. Penyakit ini umum terjadi pada tanaman jagung di berbagai daerah dengan iklim yang hangat dan lembap. Gejalanya meliputi adanya bercak-bercak berwarna kuning atau oranye pada daun tanaman jagung. Infeksi karat jagung biasanya tidak menyebabkan kerusakan yang serius pada hasil panen, tetapi dapat mengurangi pertumbuhan dan produktivitas tanaman jika serangan parah terjadi.")
st.markdown("2. Corn Gray Leaf Spot atau bercak daun abu-abu pada jagung disebabkan oleh jamur Cercospora zeae-maydis. Penyakit ini biasanya terjadi pada pertengahan hingga akhir musim tanam dan lebih umum terjadi di daerah yang lembap. Gejala utamanya adalah adanya bercak-bercak berwarna abu-abu atau coklat kehitaman pada daun jagung. Serangan berat dapat menyebabkan penurunan produksi dan kualitas jagung.")
st.markdown("3. Corn Northern Leaf Blight atau bercak daun utara pada jagung disebabkan oleh jamur Exserohilum turcicum. Penyakit ini biasanya terjadi pada musim panas yang lembap dan hangat. Gejalanya meliputi adanya bercak-bercak berwarna coklat atau hijau keabu-abuan pada daun tanaman jagung. Serangan yang parah dapat menyebabkan kerusakan pada daun, mengurangi efisiensi fotosintesis, dan berpotensi mengurangi hasil panen.")
st.markdown("4. Corn Healty adalah kondisi bahwa tanaman jagung anda dalam kondisi sehat.")
