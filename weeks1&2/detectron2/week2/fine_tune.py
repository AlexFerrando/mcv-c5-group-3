from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader, detection_utils, DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import os
import wandb
import cv2
import albumentations as A
import torch
import numpy as np
import consts

from read_data_help import CustomKittiMotsDataset
from detector import Detector

# WandB Project Name
WANDB_PROJECT_NAME = "C5-Week2"
DATASET_NAME = 'kitti_mots'

class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, augmentations=None):
        """Initializes albumentations mapper.

        Args:
            cfg (Any): Configuration for the model.
            is_train (bool, optional): Whether is train dataset. Defaults to True.
            augmentations (Any, optional): Augmentations from albumentations to apply. Defaults to None.
        """
        super().__init__(cfg, is_train, instance_mask_format="bitmask")
        self.augmentations = augmentations
        self.mask_format = cfg.INPUT.MASK_FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = cv2.imread(dataset_dict["file_name"]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_train and "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            bboxes = [obj["bbox"] for obj in annotations]
            category_ids = [obj["category_id"] for obj in annotations]
            
            transformed = self.augmentations(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed["image"]
            
            # Update the bounding boxes with transformed coordinates
            for i, annotation in enumerate(annotations):
                if i < len(transformed["bboxes"]):
                    annotation["bbox"] = transformed["bboxes"][i]
            
            # Convert to Instances format for Detectron2
            annos = []
            for annotation in annotations:
                obj = {
                    "bbox": annotation["bbox"],
                    "bbox_mode": annotation.get("bbox_mode", BoxMode.XYWH_ABS),
                    "segmentation": annotation.get("segmentation"),
                    "category_id": annotation["category_id"],
                    "iscrowd": annotation.get("iscrowd", 0),
                }
                annos.append(obj)
            
            # Create Instances object with the correct image size
            instances = detection_utils.annotations_to_instances(annos, image.shape[:2], mask_format=self.mask_format)
            dataset_dict["instances"] = instances
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))
        
        return dataset_dict

def get_augmentations() -> A.Compose:
	"""Get the augmentations to apply.

	Returns:
		A.Compose: Compose of augmentations from albumentations.
	"""
	return A.Compose([
		A.MotionBlur(p=0.25, blur_limit=(3, 8)),
  		A.Illumination(p=0.4, intensity_range=(0.1, 0.2)),
        A.AtLeastOneBBoxRandomCrop(p=0.2, height=185, width=613),
        A.Rotate(p=0.3, limit=(-5, 5))
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=100, min_visibility=0.1, label_fields=['category_ids']))


class WandbTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AlbumentationsMapper(cfg, is_train=True, augmentations=get_augmentations())
        return build_detection_train_loader(cfg, mapper=mapper)

def train_model():

    wandb.init(project=WANDB_PROJECT_NAME, name="Fine-tuning")

    detector = Detector()

    detector.cfg.INPUT.MASK_FORMAT = "bitmask"
    detector.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    detector.cfg.DATASETS.TRAIN = ("kitti_mots_train",)
    detector.cfg.DATASETS.TEST = ("kitti_mots_val",)
    detector.cfg.DATALOADER.NUM_WORKERS = 2

    # Use WandB hyperparameters
    detector.cfg.SOLVER.IMS_PER_BATCH = wandb.config.images_per_batch
    detector.cfg.SOLVER.BASE_LR = wandb.config.learning_rate
    detector.cfg.SOLVER.MAX_ITER = wandb.config.max_iter
    detector.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = wandb.config.batch_size_per_image
    detector.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    os.makedirs(detector.cfg.OUTPUT_DIR, exist_ok=True)

    print('Starting training...')
    trainer = WandbTrainer(detector.cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print('Evaluating trained model...')
    eval_cfg = detector.cfg.clone()
    eval_cfg.MODEL.WEIGHTS = os.path.join(detector.cfg.OUTPUT_DIR, "model_final.pth")
    eval_predictor = DefaultPredictor(eval_cfg)

    evaluator = COCOEvaluator(DATASET_NAME + "_val", eval_cfg, False, output_dir="evaluation")
    val_loader = build_detection_test_loader(eval_cfg, DATASET_NAME + "_val")

    print("Running inference...")
    eval_results = inference_on_dataset(eval_predictor.model, val_loader, evaluator)
    print(eval_results)

    segm_results = eval_results.get("segm", {})

    wandb.log({
        "test_segm_AP": segm_results.get("AP", 0.0),
        "test_segm_AP50": segm_results.get("AP50", 0.0),
        "test_segm_AP75": segm_results.get("AP75", 0.0),
        "test_segm_loss": 1 - segm_results.get("AP", 0.0)  # example of a "loss-like" metric
    })
    
    wandb.finish()

def run_hyperparameter_search(count=10):
    """Runs WandB hyperparameter tuning using random search."""
    
    sweep_config = {
        'name': 'Detectron2',
        'method': 'random',
        'metric': {
            'name': 'validation_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.001},
            'batch_size_per_image': {'values': [32, 64, 128]},
            'images_per_batch': {'values': [2, 4, 8]},
            'max_iter': {'values': [1500, 3000, 6000]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, function=train_model, count=count)

if __name__ == "__main__":

    for split in ["train", "val"]:
        DatasetCatalog.register(DATASET_NAME + "_" + split, 
                                lambda: CustomKittiMotsDataset(consts.KITTI_MOTS_PATH, use_coco_ids=False, split=split))
        MetadataCatalog.get(DATASET_NAME + "_" + split).set(thing_classes=["0", "1"])

    wandb.login(key='51ba71e69bc476d1171b9ff36193ad111db031f5')

    run_hyperparameter_search(count=10)
