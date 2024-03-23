import streamlit as st
from PIL import Image
import base64

# Menambahkan gambar latar belakang dari file lokal
def set_background(image):
    encoded_image = base64.b64encode(image).decode()
    background = f'''
    <style>
    .stApp {{
        background-image: url('data:image/jpg;base64,{encoded_image}');
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(background, unsafe_allow_html=True)

# Load background image
background_image = Image.open("jg2.jpg")

# Set background
set_background(background_image.tobytes())

# Sidebar
st.sidebar.title("Corn Disease Detection")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Main content
st.title("Welcome to Corn Disease Detection")

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

    # Display image with prediction
    st.image(test_image, caption=f"Prediction: {hasil_prediksi}", use_column_width=True)
else:
    st.info("Please upload an image to perform the prediction.")
