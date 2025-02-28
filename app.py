import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
import h5py
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.transform import resize
from PIL import Image

# Streamlit UI
st.title("WatNet - Surface Water Detection")

# User uploads the model file
model_file = st.file_uploader("Upload TensorFlow Model (.h5)", type=["h5"])

# Upload multi-band TIFF image
image_file = st.file_uploader("Upload Multi-band TIFF Image", type=["tif", "tiff"])

# Initialize model and image variables
model = None
image = None

# Load the model
if model_file:
    try:
        model = load_model(model_file, compile=False)
        st.success("Model Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Read and display the TIFF image
if image_file:
    try:
        with rasterio.open(image_file) as src:
            image = src.read()  # shape (bands, height, width)
        st.success("Image Uploaded Successfully!")
        st.write(f"Image Shape: {image.shape}")

        # Show a preview of the first band
        fig, ax = plt.subplots()
        ax.imshow(image[0], cmap="gray")  # Display the first band
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading image: {e}")

# Prediction Button
if st.button("Load Model and Predict"):
    if model is not None and image is not None:
        try:
            # Resize image to match model input shape
            input_size = (256, 256)  # Adjust based on your model's expected input
            image_resized = np.array([resize(image[0], input_size)])  # Only using the first band
            
            # Normalize image
            image_resized = image_resized / 255.0

            # Make prediction
            prediction = model.predict(image_resized[np.newaxis, :, :, np.newaxis])  # Add batch and channel dimension
            st.success("Prediction completed!")

            # Display the output
            fig, ax = plt.subplots()
            ax.imshow(prediction[0].squeeze(), cmap="jet")  # Show prediction result
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please upload both the model and image before predicting!")

