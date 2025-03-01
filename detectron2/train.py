from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os
import random
import torch
import wandb
import albumentations as A
import cv2
import numpy as np

from read_data import register_kitti_mots
import consts

def random_search(count=10):
    sweep_config = {
        'name': 'Detectron2 fine-tuning',
        'method': 'random',
        'metric': {
            'name': 'validation_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'uniform',
                'min': 0.0001,
                'max': 0.001
            },
            'batch_size_per_image': {
                'values': [32, 64, 128]
            },
            'images_per_batch': {
                'values': [2, 4, 8]
            },
            'max_iter': {
                'values': [3000, 5000, 7000]
            }
        }
    }

    # Register training and validation datasets
    for d in ["train", "validation"]:
        DatasetCatalog.register(consts.KITTI_MOTS_PATH + d, lambda d=d: register_kitti_mots(consts.KITTI_MOTS_PATH, d))
        MetadataCatalog.get(consts.KITTI_MOTS_PATH + d).set(thing_classes=["Car", "Pedestrian"])

    # Create a sweep and start the agent
    sweep_id = wandb.sweep(sweep_config, project="c5_week2")
    wandb.agent(sweep_id, function=fine_tune, count=count)

def fine_tune():
    wandb.init()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Use Hyperparameters from W&B Sweep
    cfg.SOLVER.BASE_LR = wandb.config.learning_rate
    cfg.SOLVER.IMS_PER_BATCH = wandb.config.images_per_batch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = wandb.config.batch_size_per_image
    cfg.SOLVER.MAX_ITER = wandb.config.max_iter

    cfg.DATASETS.TRAIN = (f"{consts.KITTI_MOTS_PATH}train",)
    cfg.DATASETS.TEST = (f"{consts.KITTI_MOTS_PATH}validation",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.TEST.EVAL_PERIOD = 500  # Evaluate the model periodically

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize the trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Start training
    trainer.train()

class AlbumentationsMapper:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.RandomGamma(p=0.2),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        bboxes = [ann["bbox"] for ann in dataset_dict["annotations"]]
        bboxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        category_ids = [ann["category_id"] for ann in dataset_dict["annotations"]]

        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)

        for i, ann in enumerate(dataset_dict["annotations"]):
            if i < len(transformed["bboxes"]):
                x1, y1, x2, y2 = transformed["bboxes"][i]
                ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]

        dataset_dict["image"] = torch.as_tensor(transformed["image"].transpose(2, 0, 1))
        return dataset_dict


if __name__ == "__main__":
    random_search(count=10)