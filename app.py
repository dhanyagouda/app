import streamlit as st
import tensorflow as tf
import numpy as np
import rasterio
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import os
import tempfile
import matplotlib.pyplot as plt
from skimage.transform import resize

# Show TensorFlow Version
st.write(f"TensorFlow Version: {tf.__version__}")

# Streamlit UI
st.title("Semantic Segmentation with TensorFlow")

# Upload model file
model_file = st.file_uploader("Upload TensorFlow Model (.h5)", type=["h5"])

# Upload TIFF file
image_file = st.file_uploader("Upload Multi-band TIFF Image", type=["tif", "tiff"])

if model_file and image_file:
    try:
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file:
            temp_model_file.write(model_file.read())
            temp_model_path = temp_model_file.name

        # Load the model with custom_objects
        model = load_model(temp_model_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2D}, compile=False)
        st.success("Model Loaded Successfully!")

        # Read the uploaded TIFF image
        with rasterio.open(image_file) as src:
            image = src.read()  # shape (bands, height, width)

        # Transpose to (height, width, bands)
        image = image.transpose(1, 2, 0)

        # Normalize the image
        image = image.astype('float32') / 255.0

        # Resize to model's input size (assuming 512x512)
        resized_img = resize(image, (512, 512, image.shape[2]), anti_aliasing=True)

        # Expand dimensions for model input
        input_batch = np.expand_dims(resized_img, axis=0)

        # Make prediction
        pred_mask = model.predict(input_batch)[0, :, :, 0]

        # Convert to binary mask
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Display mask
        st.subheader("Predicted Mask")
        fig, ax = plt.subplots()
        ax.imshow(binary_mask, cmap='viridis')
        ax.axis('off')
        st.pyplot(fig)

        # Allow downloading the mask
        st.download_button("Download Mask", binary_mask.tobytes(), "predicted_mask.png", "image/png")

        # Clean up temp file
        os.remove(temp_model_path)

    except Exception as e:
        st.error(f"Error: {e}")
