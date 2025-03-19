import streamlit as st
import requests
from PIL import Image
import io

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/predict/"

st.title("Weed Detection Demo")

# Create two columns for side-by-side images
col1, col2 = st.columns(2)

# Placeholder images before user uploads one
original_image_placeholder = col1.empty()
annotated_image_placeholder = col2.empty()

# Upload file section below images
uploaded_file = st.file_uploader("Upload an image for inference", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load uploaded image
    image = Image.open(uploaded_file)
    
    # Display original image on the left
    original_image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Send image to FastAPI for inference
    with st.spinner("Running inference..."):
        response = requests.post(API_URL, files={"file": img_bytes})

    if response.status_code == 200:
        annotated_image = Image.open(io.BytesIO(response.content))

        # Display annotated image on the right
        annotated_image_placeholder.image(annotated_image, caption="Annotated Image", use_column_width=True)
    else:
        st.error("Error processing image.")
