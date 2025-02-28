
import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize

# Streamlit UI
st.title("Semantic Segmentation with TensorFlow")

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

    # Transpose to (height, width, bands)
    image = image.transpose(1, 2, 0)

    # Normalize
    image = image.astype('float32') / 255.0

    # Resize if necessary (assuming model expects 512x512)
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

    # Option to download mask
    st.download_button("Download Mask", binary_mask.tobytes(), "predicted_mask.png", "image/png")
