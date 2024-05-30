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
st.markdown("*Aplikasi ini dapat membantu dalam mengklasifikasi kondisi tanaman jagung anda*")

#add image
# Define image size
IMG_SIZE = (299, 299)
st.image(image="ss.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

#penjelasan
st.markdown("Aplikasi ini berguna untuk mendeteksi penyakit pada tanaman jagung menggunakan teknologi kecerdasan buatan (AI) dan algoritma deep learning untuk mendiagnosis penyakit pada tanaman jagung melalui gambar yang diunggah ke aplikasi ini. Dataset yang digunakan dalam sistem ini terdiri dari ribuan gambar tanaman jagung yang terinfeksi penyakit dan sehat. Saat pengguna mengunggah gambar tanaman jagung, sistem akan menganalisis gambar tersebut dan memberikan diagnosis. Algoritma deep learning digunakan karena dapat mempelajari fitur-fitur kompleks yang terkait dengan penyakit pada tanaman jagung dan menghasilkan diagnosis yang lebih akurat.")

st.markdown(":corn: Terdapat 4 jenis kategori yang aplikasi dapat deteksi yaitu Corn Common Rust, Corn Northern Leaf Blight, Corn Gray Leaf Spot, dan Corn Healty yang akan di proses dibawah ini : ")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict
  
    test_image = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.expand_dims(test_image, 0)


# Fungsi untuk melakukan prediksi
def predict_disease(model, img_array, corndiseases_classes, threshold=0.5):
    # Melakukan prediksi dengan model
    predictions = model.predict(img_array)
    
    # Mendapatkan probabilitas tertinggi dan indeksnya
    max_prob = np.max(predictions[0])
    max_index = np.argmax(predictions[0])
    
    # Mengecek apakah probabilitas tertinggi lebih besar dari threshold
    if max_prob >= threshold:
        hasil_prediksi = corndiseases_classes[max_index]
    else:
        hasil_prediksi = "data tidak sesuai"
    
    return hasil_prediksi

# Memuat model yang telah dilatih
model = tf.keras.models.load_model('path_to_your_model.h5')

# Misalnya img_array adalah array gambar input yang sudah dipreproses
# Anda perlu menggantinya dengan proses preproses yang sesuai dengan model Anda
# Contoh: img_array = preprocess_image('path_to_image.jpg')

# Misalnya corndiseases_classes adalah daftar nama kelas penyakit jagung
corndiseases_classes = ['Class1', 'Class2', 'Class3']  # Sesuaikan dengan kelas Anda

# Pastikan img_array adalah numpy array dan memiliki shape yang benar
# Contoh: img_array = np.expand_dims(img_array, axis=0)  # Jika input adalah gambar tunggal

# Memanggil fungsi prediksi
hasil_prediksi = predict_disease(model, img_array, corndiseases_classes)
print(hasil_prediksi)

#---
st.subheader(" Penjelasan mengenai jenis-jenis penyakit pada tanaman jagung ")

st.markdown("1.Corn Common Rust atau karat jagung adalah penyakit yang disebabkan oleh jamur Puccinia sorghi. Penyakit ini umum terjadi pada tanaman jagung di berbagai daerah dengan iklim yang hangat dan lembap. Gejalanya meliputi adanya bercak-bercak berwarna kuning atau oranye pada daun tanaman jagung. Infeksi karat jagung biasanya tidak menyebabkan kerusakan yang serius pada hasil panen, tetapi dapat mengurangi pertumbuhan dan produktivitas tanaman jika serangan parah terjadi.")
st.markdown("2.Corn Gray Leaf Spot atau bercak daun abu-abu pada jagung disebabkan oleh jamur Cercospora zeae-maydis. Penyakit ini biasanya terjadi pada pertengahan hingga akhir musim tanam dan lebih umum terjadi di daerah yang lembap. Gejala utamanya adalah adanya bercak-bercak berwarna abu-abu atau coklat kehitaman pada daun jagung. Serangan berat dapat menyebabkan penurunan produksi dan kualitas jagung.")
st.markdown("3.Corn Northern Leaf Blight atau bercak daun utara pada jagung disebabkan oleh jamur Exserohilum turcicum. Penyakit ini biasanya terjadi pada musim panas yang lembap dan hangat. Gejalanya meliputi adanya bercak-bercak berwarna coklat atau hijau keabu-abuan pada daun tanaman jagung. Serangan yang parah dapat menyebabkan kerusakan pada daun, mengurangi efisiensi fotosintesis, dan berpotensi mengurangi hasil panen.")
st.markdown("4.Corn Healty adalah kondisi bahwa tanaman jagung anda dalam kondisi sehat.")
