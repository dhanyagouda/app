import streamlit as st
import numpy as np
import rasterio
import tensorflow as tf
import cv2
import tempfile
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.transform import resize
from rasterio.io import MemoryFile
from PIL import Image

# Streamlit UI
st.title("WatNet")

# Upload model file
model_file = st.file_uploader("Upload TensorFlow Model (.h5)", type=["h5"])
image_file = st.file_uploader("Upload Multi-band TIFF Image", type=["tif", "tiff"])

if model_file and image_file:
    # Load the model correctly
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model:
        temp_model.write(model_file.read())
        model_path = temp_model.name
    model = load_model(model_path, compile=False)
    st.success("Model Loaded Successfully!")

    # Read the TIFF image correctly
    with MemoryFile(image_file.read()) as memfile:
        with memfile.open() as src:
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

    # Convert mask to PNG
    mask_image = Image.fromarray(binary_mask)
    img_bytes = io.BytesIO()
    mask_image.save(img_bytes, format="PNG")

    # Option to download mask
    st.download_button("Download Mask", img_bytes.getvalue(), "predicted_mask.png", "image/png")
