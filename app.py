import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import base64

# Set default values for prediction result and predicted image
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
st.markdown("*This application can help classify the condition of your corn plants*")

# Add image
# Define image size
IMG_SIZE = (299, 299)
st.image(image="ss.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Explanation
st.markdown("This application is useful for detecting diseases in corn plants using artificial intelligence (AI) technology and deep learning algorithms to diagnose diseases in corn plants through images uploaded to this application. The dataset used in this system consists of thousands of images of corn plants infected with diseases and healthy ones. When users upload an image of a corn plant, the system will analyze the image and provide a diagnosis. Deep learning algorithms are used because they can learn complex features related to diseases in corn plants and produce more accurate diagnoses.")

st.markdown(":corn: There are 4 categories of diseases that the application can detect, namely Corn Common Rust, Corn Northern Leaf Blight, Corn Gray Leaf Spot, and Corn Healthy which will be processed below: ")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Check if the uploaded image meets the expected dimensions
    test_image = Image.open(uploaded_file)
    if test_image.size != IMG_SIZE:
        st.error("Uploaded image does not match the required dimensions. Please upload an image with dimensions (299, 299).")
        hasil_prediksi = "Image does not match"
    else:
        # Predict
        test_image = test_image.resize(IMG_SIZE)
        img_array = np.expand_dims(test_image, 0)

        predictions = model.predict(img_array)
        hasil_prediksi = corndiseases_classes[np.argmax(predictions[0])]

    # Display result
    st.success(f"Prediction: {hasil_prediksi}")

st.subheader("Explanation of the types of diseases in corn plants")

st.markdown("1. Corn Common Rust is a disease caused by the fungus Puccinia sorghi. This disease commonly occurs on corn plants in various regions with warm and humid climates. Symptoms include yellow or orange spots on corn plant leaves. Common rust infection usually does not cause serious damage to crops, but it can reduce plant growth and productivity if severe infestations occur.")
st.markdown("2. Corn Gray Leaf Spot, caused by the fungus Cercospora zeae-maydis. This disease usually occurs in mid to late growing season and is more common in humid areas. The main symptoms are gray or dark brown spots on corn leaves. Severe attacks can cause yield losses and decrease corn quality.")
st.markdown("3. Corn Northern Leaf Blight, caused by the fungus Exserohilum turcicum. This disease usually occurs in warm and humid summers. Symptoms include brown or grayish-green spots on corn plant leaves. Severe attacks can cause leaf damage, reduce photosynthesis efficiency, and potentially reduce yields.")
st.markdown("4. Corn Healthy indicates that your corn plants are in a healthy condition.")

