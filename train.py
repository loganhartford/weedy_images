from ultralytics import YOLO

# datapath = "coco8.yaml"
datapath = "D:\Documents\GitHub\weedy_images\datasets\oct16-augmented\data.yaml"

if __name__ == '__main__':
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=datapath, epochs=1, device=0)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    # results = model("https://ultralytics.com/images/bus.jpg")

    # # Export the model to ONNX format
    success = model.export(format="ncnn")