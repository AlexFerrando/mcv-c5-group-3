from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import os
import wandb
import albumentations as A
import consts

from utils import register_datasets

# WandB Project Name
WANDB_PROJECT_NAME = "C5-Week1"

class AlbumentationsMapper(DatasetMapper):
    def __init__(self):
        self.augmentations = self.get_augmentations()

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        
        return dataset_dict
    
    @staticmethod
    def get_augmentations():
        return A.Compose([
            A.RandomCrop(width=800, height=290),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(p=0.25),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category']))


class WandbTrainer(DefaultTrainer):
    def __init__(self, cfg):
        wandb.init(project=WANDB_PROJECT_NAME, config=cfg)
        super().__init__(cfg)

def train_model():

    wandb.init(project=WANDB_PROJECT_NAME, name="Detectron2")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(consts.MODEL_NAME))

    cfg.DATASETS.TRAIN = ("kitti_mots_train",)
    cfg.DATASETS.TEST = ("kitti_mots_validation",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(consts.MODEL_NAME)

    # Use WandB hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = wandb.config.images_per_batch
    cfg.SOLVER.BASE_LR = wandb.config.learning_rate
    cfg.SOLVER.MAX_ITER = wandb.config.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = wandb.config.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Car & Pedestrian

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print('Starting training...')
    trainer = WandbTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print('Evaluating trained model...')
    eval_cfg = cfg.clone()
    eval_cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    eval_predictor = DefaultPredictor(eval_cfg)

    dataset_name = ''
    evaluator = COCOEvaluator(dataset_name, eval_cfg, False, output_dir="evaluation/")
    val_loader = build_detection_test_loader(eval_cfg, dataset_name)
    print(inference_on_dataset(eval_predictor.model, val_loader, evaluator))
    
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

    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, function=train_model, count=count)

if __name__ == "__main__":
    wandb.login(key='51ba71e69bc476d1171b9ff36193ad111db031f5')

    register_datasets(consts.KITTI_MOTS_PATH)

    run_hyperparameter_search(count=5)
