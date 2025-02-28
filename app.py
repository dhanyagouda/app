import streamlit as st
import tensorflow as tf
import rasterio
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO

st.title("Upload TensorFlow Model and Image for Processing")

# Upload model file
model_file = st.file_uploader("Upload TensorFlow Model (.h5)", type=["h5"])

# Upload TIFF file
image_file = st.file_uploader("Upload Multi-band TIFF Image", type=["tif", "tiff"])

if model_file and image_file:
    try:
        # Load model from uploaded file (Fix: Read as BytesIO)
        with BytesIO(model_file.read()) as f:
            model = load_model(f, compile=False)
        st.success("Model Loaded Successfully!")

        # Read TIFF image
        with rasterio.open(image_file) as src:
            image = src.read()  # shape (bands, height, width)
        
        st.success("Image Loaded Successfully!")

    except Exception as e:
        st.error(f"Error: {str(e)}")
