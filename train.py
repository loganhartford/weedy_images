from ultralytics import YOLO

datapath = "coco8.yaml"
# datapath = "D:\Documents\GitHub\weedy_images\datasets\oct16-augmented\data.yaml"

if __name__ == '__main__':
    # Load pre-trained model
    model = YOLO("yolo11n.pt")
    results = model.train(data=datapath, epochs=3, device=0)

    # Validation
    results = model.val()

    # Inference
    # results = model("https://ultralytics.com/images/bus.jpg")

    success = model.export(format="ncnn")