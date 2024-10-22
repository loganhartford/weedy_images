from ultralytics import YOLO
import os
import csv

save_dir = "models/saves"
results_file = "augmentation_results.csv"

def get_next_save_filename(base_name="model", extension=".pt", directory=save_dir):
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    existing_saves = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
    
    if existing_saves:
        save_numbers = [int(f[len(base_name):-len(extension)]) for f in existing_saves if f[len(base_name):-len(extension)].isdigit()]
        next_number = max(save_numbers) + 1 if save_numbers else 1
    else:
        next_number = 1

    return os.path.join(directory, f"{base_name}{next_number}{extension}")

# Save results to CSV file
def save_results_to_csv(params, metrics):
    file_exists = os.path.isfile(results_file)
    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = ['parameter', 'value', 'mAP50', 'mAP75', 'mAP50-95']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'parameter': params['parameter'],
            'value': params['value'],
            'mAP50': metrics.box.map50,
            'mAP75': metrics.box.map75,
            'mAP50-95': metrics.box.map
        })

# Parameter variations to test
param_variations = [
    {"parameter": "hsv_h", "values": [0.01, 0.015, 0.9]},
    {"parameter": "hsv_s", "values": [0.01, 0.7, 0.9]},
    {"parameter": "hsv_v", "values": [0.01, 0.4, 0.9]},
    {"parameter": "degrees", "values": [0.0, 5.0, 20.0]},
    {"parameter": "scale", "values": [0.1, 0.5, 0.7]},
    {"parameter": "shear", "values": [-20, 0, 30]},
    {"parameter": "perspective", "values": [0.0, 0.0005, 0.001]},
    {"parameter": "flipud", "values": [0, 0.5, 0.9]},
    {"parameter": "fliplr", "values": [0, 0.5, 0.9]},
    {"parameter": "bgr", "values": [0, 0.5, 0.9]},
    {"parameter": "mosaic", "values": [0, 0.5, 0.9]},
    {"parameter": "mixup", "values": [0, 0.5, 0.9]},
    {"parameter": "copy_paste", "values": [0, 0.5, 0.9]},
    {"parameter": "auto_augment", "values": ["randaugment", "autoaugment", "augmix"]},
    {"parameter": "erasing", "values": [0, 0.5, 0.9]},
    {"parameter": "crop_fraction", "values": [0, 0.5, 0.9]}
]

datapath = "coco8.yaml"
# datapath = "D:\Documents\GitHub\weedy_images\datasets\oct-16-unaugmented\data.yaml"
# datapath = "D:\Documents\GitHub\weedy_images\datasets\oct16-augmented\data.yaml"
model_path = "models/yolo11n.pt"

if __name__ == '__main__':
    # Load pre-trained model
    model = YOLO(model_path)
    
    for param_set in param_variations:
        param_name = param_set["parameter"]
        for value in param_set["values"]:
            print(f"Training with {param_name}={value}")
            
            try:
                results = model.train(
                    data=datapath,
                    epochs=1,
                    device=0,
                    # Use the current parameter for this run
                    **{param_name: value},
                )

                # Validation
                metrics = model.val()
                print(f"Validation metrics for {param_name}={value}: mAP50: {metrics.box.map50}, mAP75: {metrics.box.map75}, mAP50-95: {metrics.box.map}")
                
                # Save results to CSV
                save_results_to_csv({"parameter": param_name, "value": value}, metrics)

            except Exception as e:
                print(f"Training interrupted or failed due to: {e}")
            
            finally:
                # Save the model state
                save_path = get_next_save_filename(base_name=f"{param_name}_{value}")
                print(f"Saving model trained with {param_name}={value}...")
                model.save(save_path)
                print("Model saved successfully.")
