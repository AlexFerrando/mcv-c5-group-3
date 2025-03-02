from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import json
import os
from tqdm import tqdm

from detector import Detector
import consts
from read_data import register_kitti_mots 

def evaluate_model(dataset_name, detector):
    """
    Evaluate the model using COCO metrics (AP, IoU) on the KITTI MOTS dataset.
    
    Args:
        dataset_name (str): Dataset name for testing (e.g., "kitti_mots_testing").
        detector (Detector): The initialized Detector instance.
    """  
    print("Starting model evaluation...")

    # Create COCO evaluator
    evaluator = COCOEvaluator(dataset_name, output_dir="output/")
    
    # Build detection test dataloader
    val_loader = build_detection_test_loader(detector.cfg, dataset_name)
    
    # Perform evaluation
    results = inference_on_dataset(detector.predictor.model, 
                                   tqdm(val_loader, desc="Evaluting..."), evaluator)
    print("Full evaluation results:", results)

    # Save results
    save_evaluation_results(results)
    print("Evaluation completed. Results saved in './output/evaluation_results.json'")


def save_evaluation_results(results):
    """
    Save the evaluation results to a JSON file.
    
    Args:
        results (dict): The evaluation results dictionary.
    """
    if "bbox" not in results:
        print("Warning: 'bbox' key not found in results! No bounding box evaluations were computed.")
        print("Available keys in results:", results.keys())
        results["bbox"] = {
            "AP50": 0.0,
            "AP75": 0.0,
            "AR50": 0.0,
            "AR75": 0.0,
        }

    # Extract metrics
    evaluation_metrics = {
        "AP@IoU=0.50": results["bbox"].get("AP50", 0.0),
        "AP@IoU=0.75": results["bbox"].get("AP75", 0.0),
        "AR@IoU=0.50": results["bbox"].get("AR50", 0.0),
        "AR@IoU=0.75": results["bbox"].get("AR75", 0.0),
    }
    
    # Save to a JSON file
    output_path = "./output/evaluation_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    
    print(f"Evaluation results saved to {output_path}")


if __name__ == '__main__':

    # Register dataset
    dataset_path = consts.KITTI_MOTS_PATH
    register_kitti_mots(dataset_path)
    dataset_name = "kitti_mots_testing"

    # Initialize Detector (ensuring model is loaded only once)
    detector = Detector()

    # Evaluate the model
    evaluate_model(dataset_name, detector)