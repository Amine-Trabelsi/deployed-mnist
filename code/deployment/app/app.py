import streamlit as st
import numpy as np
import requests
from PIL import Image

# Set up the page title and header
st.title("MNIST Digit Classifier")
st.header("Upload a handwritten digit image")

# Define a function to send the image to the FastAPI for prediction
def predict(image):
    url = "http://api:8000/predict"
    data = {"image": image.flatten().tolist()}
    response = requests.post(url, json=data)
    return response.json()

# Image input from the user
uploaded_file = st.file_uploader("Choose a digit image (28x28)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the image to grayscale and reshape it
    image = np.array(Image.open(uploaded_file).convert('L').resize((28, 28)))

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button('Predict'):
        result = predict(image)
        st.write(f"Predicted Digit: {result['prediction']}")
