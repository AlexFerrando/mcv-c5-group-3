import os
import argparse

from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

import consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model", default="yolo11n-seg.pt")
    parser.add_argument("--output_path", type=str, help="Path to save the results", default="C5-yolo-segmentation")
    args = parser.parse_args()

    wandb.login(key = "8410a2da3f323633210ca8d25ce6862368d9f489")
    model = YOLO(args.model_path)
    config_dataset_path = os.path.join(consts.PATH_KITTI_MOTS_YOLO_SEGMENTATION, "kitti_mots_config.yaml")
    add_wandb_callback(model, enable_model_checkpointing = False, enable_train_validation_logging=True)
    model.train(
        data=config_dataset_path,
        classes=consts.YOLO_CLASSES,
        
        epochs=30,
        optimizer="auto",
        freeze=None,
        
        lr0=0.01,
        warmup_epochs=3,
        mask_ratio=4,
        batch=0.4,
        
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        erasing=0.4,
        crop_fraction=1.0,
        # auto_augment='autoaugment'
        
        patience=20,
        save_period=50,
        
        plots=False,
        device='cuda',
        project=args.output_path,
        name=f"default yolo segment",
    )
