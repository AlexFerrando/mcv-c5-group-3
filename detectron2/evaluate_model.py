from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo

import torch
import json

import consts
from read_data import register_kitti_mots 

def evaluate_model(dataset_name):
    """
    Evaluate the model using COCO metrics (AP, IoU) on the KITTI MOTS dataset.
    Args:
        dataset_path (str): Path to the KITTI MOTS dataset.
    """    
    # Load configuration from Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(consts.MODEL_NAME))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(consts.MODEL_NAME)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for detection
    
    # Load model for evaluation
    predictor = DefaultPredictor(cfg)
    
    # Create COCO evaluator for the testing dataset
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="output/")
    
    # Perform inference and evaluate using COCO metrics
    dataset_dicts = DatasetCatalog.get(dataset_name)
    results = inference_on_dataset(predictor.model, dataset_dicts, evaluator)
    
    # Save results to a JSON file
    save_evaluation_results(results)


def save_evaluation_results(results):
    """
    Save the evaluation results to a JSON file.
    Args:
        results (dict): The evaluation results dictionary.
    """
    # Define the output path
    output_path = "output/evaluation_results.json"
    
    # Average Precision (AP) & Average Recall (AR)
    evaluation_metrics = {
        "AP@0.50": results["bbox"]["AP"],       # Standard AP metric
        "AP@0.75": results["bbox"]["AP75"],     # High IoU threshold
        "AR@0.50": results["bbox"]["AR"],       # How many GT objects are TP
        "AR@0.75": results["bbox"]["AR75"]  	# High IoU threshold
    }
    
    # Save the metrics to a JSON file
    with open(output_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    print(f"Evaluation results saved to {output_path}")


if __name__ == '__main__':

    dataset_path = consts.KITTI_MOTS_PATH
    register_kitti_mots(dataset_path)
    dataset_name = "kitti_mots_testing"

    evaluate_model(dataset_name)