import os
import argparse

from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

import consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model", default="yolo11n.pt")
    parser.add_argument("--output_path", type=str, help="Path to save the results", default="/ghome/c5mcv03/mcv-c5-group-3/outputs/pol/job_outputs")
    args = parser.parse_args()

    wandb.login(key = "8410a2da3f323633210ca8d25ce6862368d9f489")
    model = YOLO(args.model_path)
    config_dataset_path = os.path.join(consts.PATH_KITTI_MOTS_YOLO, "kitti_mots_config.yaml")
    add_wandb_callback(model, enable_model_checkpointing = True)
    model.train(
        data=config_dataset_path,
        epochs=30,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        translate=0.1,
        scale=0.25,
        fliplr=0.0,
        erasing=0.4,
        crop_fraction=0.9,
        # auto_augment='autoaugment'
        patience=10,
        lr0=0.1,
        batch=16,
        save_period=50,
        device='0',
        project=args.output_path,
        name=f"yolo11n no augmentation finetuned 30 epochs 0.1lr0",
    )
