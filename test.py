from ultralytics import YOLO
import yaml

# model_path = "D:/Documents/GitHub/weedy_images/runs/detect/train4/weights/best_ncnn_model"
# img_path = "D:/Documents/GitHub/weedy_images/datasets/oct16-augmented/test/images/captured_image_20241016_153912_jpg.rf.35894ba5438fabbbc9a7650da1e6ba62.jpg"

model_path = "D:\Documents\GitHub\weedy_images\models\indoor_pose_ncnn_model"
img_path = "D:\Documents\GitHub\weedy_images\img\oct-22-indoor\captured_image_20241022_152954.jpg"
yaml_file_path = 'D:\Documents\GitHub\weedy_images\models\indoor_pose_ncnn_model\metadata.yaml'

keys = [
    "flower",
    "base",
    "upper",
    "lower",
]

if __name__ == '__main__':
    ncnn_model = YOLO(model_path, task="pose")
    results = ncnn_model(img_path)

    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Retrieve and print the names
    if 'names' in data:
        names = data['names']

    if not results:
        print("No detections found.")
    else:
        result = results[0]
        #results.show()
        result.save()
        # print(result.keypoints[0].info())

        keypoints = []
        for i, keypoint in enumerate(result.keypoints):
            data = keypoint.data
            print(f"Keypoint data: {data}")

            keypoint_dict = {
                "name": names[0],
                "flower": tuple(round(x, 3) for x in data[i][0].tolist()),
                "base": tuple(round(x, 3) for x in data[i][1].tolist()),
                "upper": tuple(round(x, 3) for x in data[i][2].tolist()),
                "lower": tuple(round(x, 3) for x in data[i][3].tolist()),
            }

            print(keypoint_dict)

        # for box in result.boxes:
        #     xyxy = box.xyxy[0].cpu()  # Get [xmin, ymin, xmax, ymax] format
        #     conf = box.conf
        #     cls = box.cls
        #     print(f"Bounding box coordinates: {xyxy}, Confidence: {conf}, Class: {cls}")