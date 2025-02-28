import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
import h5py
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.transform import resize
from PIL import Image

# Streamlit UI
st.title("watnet")

# User enters the file paths
model_file = st.file_uploader("Upload TensorFlow Model (.h5)", type=["h5"])

# Upload TIFF file
image_file = st.file_uploader("Upload Multi-band TIFF Image", type=["tif", "tiff"])

if model_file and image_file:
    # Load the model
    model = load_model(model_file, compile=False)
    st.success("Model Loaded Successfully!")

    # Read the uploaded TIFF image
    with rasterio.open(image_file) as src:
        image = src.read()  # shape (bands, height, width)

    # Process image...
    st.success("Image uploaded successfully!")

if st.button("Load Model and Predict"):
    if model_path and image_path:
        if not os.path.exists(model_path):
