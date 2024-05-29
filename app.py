import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from werkzeug.utils import secure_filename
import base64

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
st.markdown("*This application can help in classifying the condition of your corn plants*")

# Explanation
st.markdown("This application detects diseases in corn plants using artificial intelligence (AI) technology and deep learning algorithms to diagnose diseases in corn plants through uploaded images. The dataset used in this system consists of thousands of images of corn plants infected with diseases and healthy ones. When users upload an image of a corn plant, the system will analyze the image and provide a diagnosis.")

st.markdown(":corn: There are 4 categories of corn diseases that the application can detect: Corn Common Rust, Corn Northern Leaf Blight, Corn Gray Leaf Spot, and Corn Healthy.")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict
    IMG_SIZE = (299, 299)  # Define image size
    test_image = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.expand_dims(test_image, 0)

    predictions = model.predict(img_array)
    prediction_index = np.argmax(predictions[0])
    predicted_class = corndiseases_classes[prediction_index]

    # Display result
    st.success(f"Prediction: {predicted_class}")

st.subheader("Explanation of different types of corn plant diseases")

st.markdown("1. **Corn Common Rust**: Common rust is caused by the fungus Puccinia sorghi. Symptoms include yellow or orange spots on corn leaves. While it usually doesn't cause severe damage to crops, severe infections can reduce plant growth and productivity.")
st.markdown("2. **Corn Gray Leaf Spot**: Gray leaf spot is caused by the fungus Cercospora zeae-maydis. Symptoms include gray or brownish spots on corn leaves, typically occurring mid to late season in humid regions. Severe infections can lead to yield losses.")
st.markdown("3. **Corn Northern Leaf Blight**: Northern leaf blight is caused by the fungus Exserohilum turcicum. Symptoms include brown or greenish-gray spots on corn leaves, usually appearing in warm, humid summers. Severe infections can damage leaves, reduce photosynthesis efficiency, and potentially decrease yields.")
st.markdown("4. **Corn Healthy**: Indicates that your corn plants are in a healthy condition.")
