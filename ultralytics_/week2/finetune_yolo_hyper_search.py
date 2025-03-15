import os
import argparse
import wandb
from ultralytics import YOLO
# from wandb.integration.ultralytics import add_wandb_callback  # Uncomment if needed
import consts

def training_func():
    # Initialize a wandb run; the sweep agent will populate wandb.config
    wandb.init()
    config = wandb.config

    # Use sweep-provided parameters with fallback defaults
    model_path = config.get("model_path", "yolo11n-seg.pt")
    output_path = config.get("output_path", "ultralytics_/segmentation")
    
    # Initialize YOLO model
    model = YOLO(model_path)
    
    # Path to the dataset config file
    config_dataset_path = os.path.join(consts.PATH_KITTI_MOTS_YOLO_SEGMENTATION, "kitti_mots_config.yaml")
    
    # Define static training arguments
    train_args = {
         "freeze": 21,
         "imgsz": 1024,
         "device": "cuda",
         "project": output_path,
         "classes": consts.YOLO_CLASSES,
         "optimizer": {"value": "AdamW"}
         # Repeated keys removed; adjust as necessary.
    }
    
    # Gather hyperparameters from the wandb sweep
    sweep_params = {
         "lr0": config.lr0,
         "lrf": config.lrf,
         "momentum": config.momentum,
         "weight_decay": config.weight_decay,
         "warmup_epochs": config.warmup_epochs,
         "warmup_momentum": config.warmup_momentum,
         "hsv_h": config.hsv_h,
         "hsv_s": config.hsv_s,
         "hsv_v": config.hsv_v,
         "translate": config.translate,
         "scale": config.scale,
         "fliplr": config.fliplr,
    }
    
    # Optionally, add the wandb callback to log training details automatically
    # add_wandb_callback(model, enable_model_checkpointing=False, enable_train_validation_logging=True)
    
    # Train the model using the dataset config and the combined parameters
    model.train(data=config_dataset_path, epochs=20, patience=5, batch=16, imgsz=1024, **train_args, **sweep_params)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with wandb sweep")
    parser.add_argument("--model_path", type=str, default="yolo11n-seg.pt",
                        help="Path to the YOLO model")
    parser.add_argument("--output_path", type=str, default="ultralytics_/segmentation",
                        help="Path to save the results")
    args = parser.parse_args()

    # Login to wandb (you might want to remove your key from code in practice)
    wandb.login(key="8410a2da3f323633210ca8d25ce6862368d9f489")
    
    # Define the sweep configuration as a Python dictionary
    sweep_config = {
        "method": "random",  # or "random", "grid", etc.
        "metric": {
            "name": "metrics/mAP50(B)",
            "goal": "maximize"
        },
        "parameters": {
            "lr0": {"distribution": "uniform", "min": 0.00001, "max": 0.1},
            "lrf": {"distribution": "uniform", "min": 0.1, "max": 1.0},
            "momentum": {"distribution": "uniform", "min": 0.7, "max": 0.95},
            "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.001},
            "warmup_epochs": {"distribution": "uniform", "min": 0.0, "max": 5.0},
            "warmup_momentum": {"distribution": "uniform", "min": 0.0, "max": 0.95},
            "hsv_h": {"distribution": "uniform", "min": 0.0, "max": 0.1},
            "hsv_s": {"distribution": "uniform", "min": 0.0, "max": 0.7},
            "hsv_v": {"distribution": "uniform", "min": 0.1, "max": 0.4},
            "translate": {"distribution": "uniform", "min": 0.0, "max": 0.1},
            "scale": {"distribution": "uniform", "min": 0.0, "max": 0.5},
            "fliplr": {"distribution": "uniform", "min": 0.0, "max": 0.5},
            
            # Pass in static arguments from the command line if desired.
            "model_path": {"value": args.model_path},
            "output_path": {"value": args.output_path},
            
        }
    }
    # Initialize the sweep (this returns a sweep ID)
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.output_path)
    print("Sweep ID:", sweep_id)

    # Launch the sweep agent that will call the training function for each set of hyperparameters
    wandb.agent(sweep_id, function=training_func)
