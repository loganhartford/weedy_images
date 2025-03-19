import streamlit as st
import requests
from PIL import Image
import io

st.title("Weed Detection Demo")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Send to FastAPI
    with st.spinner("Running inference..."):
        response = requests.post("http://127.0.0.1:8000/predict/", files={"file": img_bytes})

    if response.status_code == 200:
        st.success("Inference complete!")
        annotated_image = Image.open(io.BytesIO(response.content))
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)
    else:
        st.error("Error processing image.")
