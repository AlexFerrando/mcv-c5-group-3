import os
import argparse

import wandb
from ray import tune
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

import consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model", default="yolo11n-seg.pt")
    parser.add_argument("--output_path", type=str, help="Path to save the results", default="ultralytics_/segmentation")
    args = parser.parse_args()

    wandb.login(key = "8410a2da3f323633210ca8d25ce6862368d9f489")
    model = YOLO(args.model_path)
    config_dataset_path = os.path.join(consts.PATH_KITTI_MOTS_YOLO_SEGMENTATION, "kitti_mots_config.yaml")
    # add_wandb_callback(model, enable_model_checkpointing = False, enable_train_validation_logging=True)


    search_space = {"lr0": tune.uniform(0.1, 0.00001),
                        "lrf": tune.uniform(0.1, 1.0),
                        "momentum": tune.uniform(0.7, 0.95),
                        "weight_decay": tune.uniform(0.0, 0.001),
                        "warmup_epochs": tune.uniform(0.0, 5.0),
                        "warmup_momentum": tune.uniform(0.0, 0.95),
                        "hsv_h": tune.uniform(0.0, 0.1),
                        "hsv_s": tune.uniform(0.0, 0.7),
                        "hsv_v": tune.uniform(0.1, 0.4),
                        "translate": tune.uniform(0.0, 0.1),
                        "scale": tune.uniform(0.0, 0.5),
                        "fliplr": tune.uniform(0.0, 0.5),
                        }
    train_args={
        "freeze": 21, "imgsz": 1024, "device": "cuda",
        "project": args.output_path,
        "name": "default_yolo_segment_adamw_0.0005lr0",
        "classes": consts.YOLO_CLASSES,
        "freeze": 21, "imgsz": 1024,
    }
    result_grid = model.tune(data=config_dataset_path, grace_period = 6, iterations=15, use_ray=True,
                             space=search_space, **train_args)
        
