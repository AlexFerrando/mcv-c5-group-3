from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader
import detectron2.data.transforms as T

import os
import wandb
import albumentations as A
import consts

from read_data import register_kitti_mots
from utils import register_datasets

# WandB Project Name
WANDB_PROJECT_NAME = "C5-Week1"

# Augmentation Wrapper for Albumentations
class AlbumentationsMapper:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(p=0.2),
        ])

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = dataset_dict["image"]
        image = self.transform(image=image)["image"]
        dataset_dict["image"] = image
        return dataset_dict

# Custom Trainer for Detectron2 + WandB
class WandBTrainer(DefaultTrainer):
    def __init__(self, cfg):
        wandb.init(project=WANDB_PROJECT_NAME, config=cfg)
        super().__init__(cfg)

    def after_step(self):
        """Log training loss to WandB after each step."""
        loss_dict = {k: v.item() for k, v in self.storage.latest().items() if "loss" in k}
        wandb.log(loss_dict)

# Function to Train & Fine-Tune the Model
def train_model():
    # Init WandB
    wandb.init(project="C5-Week1", name="Detectron")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("kitti_mots_train",)
    cfg.DATASETS.TEST = ("kitti_mots_validation",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Use WandB hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = wandb.config.images_per_batch
    cfg.SOLVER.BASE_LR = wandb.config.learning_rate
    cfg.SOLVER.MAX_ITER = wandb.config.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = wandb.config.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Car & Pedestrian

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = WandBTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

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
            'max_iter': {'values': [3000, 5000, 7000]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="KITTI_MOTS")
    wandb.agent(sweep_id, function=train_model, count=count)

if __name__ == "__main__":
    wandb.login(key='51ba71e69bc476d1171b9ff36193ad111db031f5')

    register_datasets(consts.KITTI_MOTS_PATH)

    run_hyperparameter_search(count=5)
