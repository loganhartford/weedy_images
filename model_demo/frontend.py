import streamlit as st
import requests
from PIL import Image
import io
import os

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/predict/"
SAMPLE_IMAGES_DIR = "D:/Documents/GitHub/weedy_images/model_demo/sample_images"

st.title("Try our outdoor computer vision model!")

# Create placeholders for image display windows (always visible at the top)
col1, col2 = st.columns(2)
original_image_placeholder = col1.empty()
annotated_image_placeholder = col2.empty()

# Initialize session state for sample image selection
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# Display sample images selection widgets below the display windows
st.subheader("Select a sample image for inference")
# Get list of sample images (png, jpg, jpeg)
sample_images = [filename for filename in os.listdir(SAMPLE_IMAGES_DIR) 
                 if filename.lower().endswith((".png", ".jpg", ".jpeg"))]

# Set the number of icons per row
cols_per_row = 4
# Loop over images in groups to create a grid of clickable icons
for i in range(0, len(sample_images), cols_per_row):
    cols = st.columns(cols_per_row)
    for idx, filename in enumerate(sample_images[i:i+cols_per_row]):
        with cols[idx]:
            image_path = os.path.join(SAMPLE_IMAGES_DIR, filename)
            img = Image.open(image_path)
            # Resize for a small icon display
            icon_size = (200, 200)
            img_icon = img.resize(icon_size)
            st.image(img_icon)
            if st.button("Select", key=filename):
                st.session_state.selected_sample = image_path

# Allow user to upload an image for inference
uploaded_file = st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg"])

# Determine which image to use: uploaded file takes precedence over sample image selection
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif st.session_state.selected_sample is not None:
    image = Image.open(st.session_state.selected_sample)
else:
    image = None

if image is not None:
    # Display the selected image in the original image placeholder
    original_image_placeholder.image(image, caption="Original Image", use_container_width=True)
    
    # Convert image to bytes for sending to FastAPI
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format="JPEG")
    img_bytes = img_bytes_io.getvalue()
    
    # Send image to FastAPI for inference
    with st.spinner("Running inference..."):
        response = requests.post(API_URL, files={"file": img_bytes})
    
    if response.status_code == 200:
        annotated_image = Image.open(io.BytesIO(response.content))
        annotated_image_placeholder.image(annotated_image, caption="Model Output", use_container_width=True)
    else:
        st.error("Error processing image.")
