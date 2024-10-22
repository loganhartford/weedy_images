from ultralytics import YOLO

datapath = "coco8.yaml"  # Your dataset configuration

if __name__ == '__main__':
    # Load pre-trained model
    model = YOLO("yolo11n.pt")
    
    # Train model with data augmentation parameters (default values)
    results = model.train(
        data=datapath,
        epochs=3,
        device=0,
        # Data augmentation hyperparameters
        hsv_h=0.015,                    # Adjust hue | default: 0.015
        hsv_s=0.7,                      # Adjust saturation | default: 0.7
        hsv_v=0.4,                      # Adjust brightness (value) | default: 0.4
        degrees=0.0,                    # Rotate image | default: 0.0
        translate=0.1,                  # Translate image | default: 0.1
        scale=0.5,                      # Scale image | default: 0.5
        shear=0.0,                      # Shear image | default: 0.0
        perspective=0.0,                # Perspective warp image | default: 0.0
        flipud=0.0,                     # Flip image upside down probability | default: 0.0
        fliplr=0.5,                     # Flip image left to right probability | default: 0.5
        bgr=0.0,                        # Randomly change to BGR order | default: 0.5
        mosaic=1.0,                     # Mosaic augmentation | default: 1.0
        mixup=0.0,                      # Mixup augmentation | default: 0.0
        copy_paste=0.0,                 # Copy-paste augmentation | default: 0.0
        copy_paste_mode="flip",         # Copy-paste mode | default: "flip"
        auto_augment="randaugment",     # Auto augment policy | default: "randaugment"
        erasing=0.4,                    # Random erasing probability | default: 0.4
        crop_fraction=1.0               # Fraction of image to crop | default: 1.0
    )

    # Validation
    results = model.val()

    # Export the model
    success = model.export(format="ncnn")
