import os
import argparse

from ultralytics import YOLO
import wandb
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
    add_wandb_callback(model, enable_model_checkpointing = False, enable_train_validation_logging=False)
    model.train(
        data=config_dataset_path,
        classes=consts.YOLO_CLASSES,
        
        epochs=50,
        optimizer="AdamW",
        freeze=21,
        
        lr0=0.017162956662722863,
        lrf=0.8346365649678075,
        momentum=0.7070979831851115,
        weight_decay=0.0007452824661310931,
        warmup_epochs=4,
        warmup_momentum=0.8201039478474136,
        batch=0.4,
        
        imgsz=1024, 
        hsv_h=0.0914214125150458,
        hsv_s=0.4023838445250445,
        hsv_v=0.13262171333213696,
        
        translate=0.018423185082522452,
        scale=0.29359083344271464,
        fliplr=0.08606952789145506,
        
        mosaic=1.0,
        erasing=0.4,
        crop_fraction=1.0,
        # auto_augment='autoaugment'
        
        patience=10,
        save_period=50,
        
        plots=False,
        # device='cuda',
        project=args.output_path,
        name=f"custom_augmentations_freeze21_long_run",
    )
