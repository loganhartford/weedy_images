from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import io
import cv2
from PIL import Image
import os
import yaml
from ultralytics import YOLO

# Define paths
MODEL_PATH = "D:/Documents/GitHub/weedy_images/models/outdoor_all_nano_ncnn_model"
OUTPUT_DIR = "output"

# Load model
model = YOLO(MODEL_PATH, task="pose")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Run inference
    results = model(image)

    # If no detections, return
    if not results:
        return {"message": "No detections found"}

    result = results[0]

    valid = result.boxes.conf >= 0.5
    result.boxes = result.boxes[valid]

    # Save annotated image
    output_path = os.path.join(OUTPUT_DIR, "annotated.jpg")
    result.save(filename=output_path)

    return FileResponse(output_path, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
