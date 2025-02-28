import streamlit as st
import tensorflow as tf
import rasterio
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model

st.title("Upload TensorFlow Model and Image for Processing")

# Upload model file
model_file = st.file_uploader("Upload TensorFlow Model (.h5)", type=["h5"])

# Upload TIFF file
image_file = st.file_uploader("Upload Multi-band TIFF Image", type=["tif", "tiff"])

if model_file and image_file:
    try:
        # Save uploaded model file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model:
            temp_model.write(model_file.read())  # Write contents
            temp_model_path = temp_model.name  # Get file path

        # Load model from the saved file
        model = load_model(temp_model_path, compile=False)
        st.success("Model Loaded Successfully!")

        # Read TIFF image
        with rasterio.open(image_file) as src:
            image = src.read()  # shape (bands, height, width)
        
        st.success("Image Loaded Successfully!")

        # Cleanup: Delete temporary model file after loading
        os.remove(temp_model_path)

    except Exception as e:
        st.error(f"Error: {str(e)}")
