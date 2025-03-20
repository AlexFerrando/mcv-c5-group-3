from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import os
import wandb
import cv2
import numpy as np
from datasets import load_dataset

# Load dataset from Hugging Face
dataset = load_dataset("RationAI/PanNuke")

# Define category names (PanNuke has 5 categories: Neoplastic, Inflammatory, Connective, Dead, Epithelial)
CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

# Ensure image storage directory exists
IMAGE_DIR = os.path.abspath("./pannuke_images/")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Convert dataset to Detectron2 format
def convert_to_detectron2_format(dataset_split, split_name):
    dataset_dicts = []
    
    for idx, item in enumerate(dataset_split):
        record = {}
        
        # Convert image to NumPy array and save it locally
        image_array = np.array(item["image"])  
        image_filename = f"{split_name}_{idx}.jpg"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        cv2.imwrite(image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        height, width = image_array.shape[:2]
        record["file_name"] = image_path
        record["height"] = height
        record["width"] = width
        record["image_id"] = idx
        
        objs = []
        for obj_idx, mask_pil in enumerate(item["instances"]):  # Iterate over instance masks
            mask = np.array(mask_pil, dtype=np.uint8)  # Convert mask to NumPy array
            if mask.sum() == 0:
                continue  # Skip empty masks
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue  # Skip invalid contours
                
                contour = contour.flatten().tolist()
                x, y, w, h = cv2.boundingRect(np.array(contour).reshape(-1, 2))
                obj = {
                    "bbox": [x, y, x + w, y + h],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [contour],
                    "category_id": item["categories"][obj_idx],  # Assign correct category ID
                }
                objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

# Register the dataset with Detectron2
for split in ["fold1", "fold2"]:  # PanNuke uses 'folds' instead of train/test
    DatasetCatalog.register(f"pannuke_{split}", lambda split=split: convert_to_detectron2_format(dataset[split], split))
    MetadataCatalog.get(f"pannuke_{split}").set(thing_classes=CLASS_NAMES)

# Get metadata
metadata = MetadataCatalog.get("pannuke_fold1")

# Trainer class with WandB integration
class WandbTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

def train_model():
    wandb.init(project="PlantDoc-InstanceSegmentation", name="Mask-RCNN-PlantDoc")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pannuke_fold1",)
    cfg.DATASETS.TEST = ("pannuke_fold2",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.SOLVER.IMS_PER_BATCH = wandb.config.images_per_batch
    cfg.SOLVER.BASE_LR = wandb.config.learning_rate
    cfg.SOLVER.MAX_ITER = wandb.config.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = wandb.config.batch_size_per_image
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = WandbTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Evaluation
    eval_cfg = cfg.clone()
    eval_cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    eval_predictor = DefaultPredictor(eval_cfg)
    
    evaluator = COCOEvaluator("pannuke_fold2", eval_cfg, False, output_dir="evaluation")
    val_loader = build_detection_test_loader(eval_cfg, "pannuke_fold2")
    
    eval_results = inference_on_dataset(eval_predictor.model, val_loader, evaluator)
    print(eval_results)
    
    wandb.log({
        "test_segm_AP": eval_results.get("segm", {}).get("AP", 0.0),
        "test_segm_AP50": eval_results.get("segm", {}).get("AP50", 0.0),
        "test_segm_AP75": eval_results.get("segm", {}).get("AP75", 0.0),
        "test_segm_loss": 1 - eval_results.get("segm", {}).get("AP", 0.0)
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
            'max_iter': {'values': [1500, 3000]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="C5-Week2")
    wandb.agent(sweep_id, function=train_model, count=count)

# Run training or hyperparameter search
if __name__ == "__main__":

    wandb.login(key='51ba71e69bc476d1171b9ff36193ad111db031f5')

    run_hyperparameter_search(count=5)
