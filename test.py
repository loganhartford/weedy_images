from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# model_path = "D:/Documents/GitHub/weedy_images/runs/detect/train4/weights/best_ncnn_model"
# img_path = "D:/Documents/GitHub/weedy_images/datasets/oct16-augmented/test/images/captured_image_20241016_153912_jpg.rf.35894ba5438fabbbc9a7650da1e6ba62.jpg"

model_path = "D:/Documents/GitHub/weedy_images/runs/detect/train/weights/best.onnx"
img_path = "D:/Documents/GitHub/weedy_images/datasets/coco8/images/val/000000000036.jpg"

if __name__ == '__main__':
    ncnn_model = YOLO(model_path)
    results = ncnn_model(img_path)

    if not results:
        print("No detections found.")
    else:
        result = results[0]
        #results.show()
        result.save()

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Get [xmin, ymin, xmax, ymax] format
            conf = box.conf
            cls = box.cls
            print(f"Bounding box coordinates: {xyxy}, Confidence: {conf}, Class: {cls}")