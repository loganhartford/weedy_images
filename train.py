from ultralytics import YOLO
import os
import json
import datetime


save_dir = "models/saves"
log_file = "./training_log.json"

def get_next_save_filename(base_name="model", extension=".pt", directory=save_dir):
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    existing_saves = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
    
    if existing_saves:
        save_numbers = [int(f[len(base_name):-len(extension)]) for f in existing_saves if f[len(base_name):-len(extension)].isdigit()]
        next_number = max(save_numbers) + 1 if save_numbers else 1
    else:
        next_number = 1

    return os.path.join(directory, f"{base_name}{next_number}{extension}")

def log_dataset_and_model(dataset_path, custom_save_path, yolo_weights_path):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "dataset_path": dataset_path,
        "custom_save_path": custom_save_path,
        "yolo_weights_path": yolo_weights_path
    }

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

def train_model(datapath):
    # model = YOLO("yolo11n-pose.pt")
    model = YOLO("D:\Documents\GitHub\weedy_images\models\saves\indoor_bright.pt")
    
    try:
        # Train model with data augmentation parameters (default values)
        results = model.train(
            data=datapath,
            epochs=1000,
            device=0,
            # Model hyperparameters
            # lr0=0.005,
            # warmup_epochs=5,
            batch=32,
            cos_lr=True,
            # Data augmentation hyperparameters
            hsv_h=0.02,                    # Adjust hue | default: 0.015
            hsv_s=0.8,                      # Adjust saturation | default: 0.7
            hsv_v=0.5,                      # Adjust brightness (value) | default: 0.4
            degrees=5.0,                    # Rotate image | default: 0.0
            translate=0.1,                  # Translate image | default: 0.1
            scale=0.7,                      # Scale image | default: 0.5
            shear=0.1,                      # Shear image | default: 0.0
            perspective=0.0,                # Perspective warp image | default: 0.0
            flipud=0.0,                     # Flip image upside down probability | default: 0.0
            fliplr=0.5,                     # Flip image left to right probability | default: 0.5
            bgr=0.0,                        # Randomly change to BGR order | default: 0.5
            mosaic=1.0,                     # Mosaic augmentation | default: 1.0
            mixup=0.1,                      # Mixup augmentation | default: 0.0
            copy_paste=0.1,                 # Copy-paste augmentation | default: 0.0
            copy_paste_mode="flip",         # Copy-paste mode | default: "flip"
            auto_augment="trivialaugment",     # Auto augment policy | default: "randaugment"
            erasing=0.4,                    # Random erasing probability | default: 0.4
            crop_fraction=1.0               # Fraction of image to crop | default: 1.0
        )

        # Validation
        results = model.val()

    except Exception as e:
        print(f"Training interrupted or failed due to: {e}")
        
    finally:
        # Save the current model state if interrupted or finished
        save_path = get_next_save_filename()
        model.save(save_path)
        # log_dataset_and_model(datapath, save_path, results.save_dir)
        
    success = model.export(format="ncnn")

if __name__ == '__main__':
    paths = [
        "D:\Documents\GitHub\weedy_images\datasets\indoor-angled\data.yaml",
        # "D:\Documents\GitHub\weedy_images\datasets\indoor-all\data.yaml",
        # "D:\Documents\GitHub\weedy_images\datasets\pose-partial-outdoor\data.yaml",
    ]

    for path in paths:
        train_model(path)
    