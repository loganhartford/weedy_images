import streamlit as st
import requests
from PIL import Image
import io
import os

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/predict/"
SAMPLE_IMAGES_DIR = "D:/Documents/GitHub/weedy_images/model_demo/sample_images"

st.title("Weed Detection Demo")

# Initialize session state for sample image selection
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# Display sample images section
st.subheader("Or select a sample image")
# Get list of sample images (png, jpg, jpeg)
sample_images = [filename for filename in os.listdir(SAMPLE_IMAGES_DIR) 
                 if filename.lower().endswith((".png", ".jpg", ".jpeg"))]

# Set the number of icons per row
cols_per_row = 4
# Loop over images in groups to create a grid
for i in range(0, len(sample_images), cols_per_row):
    cols = st.columns(cols_per_row)
    for idx, filename in enumerate(sample_images[i:i+cols_per_row]):
        with cols[idx]:
            image_path = os.path.join(SAMPLE_IMAGES_DIR, filename)
            img = Image.open(image_path)
            # Resize for a small icon display
            icon_size = (100, 100)
            img_icon = img.resize(icon_size)
            st.image(img_icon)
            if st.button("Select", key=filename):
                st.session_state.selected_sample = image_path

# Allow user to upload an image for inference
uploaded_file = st.file_uploader("Upload an image for inference", type=["png", "jpg", "jpeg"])

# Determine which image to use: uploaded file takes precedence
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif st.session_state.selected_sample is not None:
    image = Image.open(st.session_state.selected_sample)
else:
    image = None

if image is not None:
    # Create two columns for side-by-side display of original and annotated images
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Input Image", use_column_width=True)
    
    # Convert image to bytes to send to the FastAPI backend
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format="JPEG")
    img_bytes = img_bytes_io.getvalue()
    
    # Send image for inference
    with st.spinner("Running inference..."):
        response = requests.post(API_URL, files={"file": img_bytes})
    
    if response.status_code == 200:
        annotated_image = Image.open(io.BytesIO(response.content))
        with col2:
            st.image(annotated_image, caption="Annotated Image", use_column_width=True)
    else:
        st.error("Error processing image.")
