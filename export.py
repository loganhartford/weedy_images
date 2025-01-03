from ultralytics import YOLO

model = YOLO("D:/Documents/GitHub/weedy_images/runs/pose/train/weights/best.pt")
model.export(format="ncnn")