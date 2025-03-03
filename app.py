import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
import h5py
import os
import cv2
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
from PIL import Image
from io import BytesIO

# Streamlit UI
st.title("watnet")
st.title("Upload TensorFlow Model and Image for Processing")

# User enters the file paths
# Upload model file
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
            st.error("Model file not found. Check the path.")
        else:
            try:
                # Validate HDF5 File
                with h5py.File(model_path, "r") as f:
                    st.success("Valid HDF5 model detected!")

                # Load model safely
                model = load_model(model_path, compile=False, safe_mode=True)
                st.success("Model Loaded Successfully!")

                # Read TIFF image
                with rasterio.open(image_path) as src:
                    image = src.read()  # shape (bands, height, width)

                # Transpose to (height, width, bands)
                image = image.transpose(1, 2, 0)

                # Normalize
                image = image.astype('float32') / 255.0

                # Resize to model's expected input shape
                resized_img = resize(image, (512, 512, image.shape[2]), anti_aliasing=True)

                # Expand dimensions for batch processing
                input_batch = np.expand_dims(resized_img, axis=0)

                # Predict the mask
                pred_mask = model.predict(input_batch)
                mask_prob = pred_mask[0, :, :, 0]
                binary_mask = (mask_prob > 0.5).astype(np.uint8) * 255

                # Display the mask
                st.subheader("Predicted Mask")
                fig, ax = plt.subplots()
                ax.imshow(binary_mask, cmap='viridis')
                ax.axis('off')
                st.pyplot(fig)

                # Convert mask to PNG
                mask_image = Image.fromarray(binary_mask)
                st.download_button("Download Mask", mask_image.tobytes(), "predicted_mask.png", "image/png")

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please enter valid file paths.")
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
