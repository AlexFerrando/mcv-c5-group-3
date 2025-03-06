from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog

import json
import os
import cv2

from detector import Detector
import consts
from read_data import register_kitti_mots 

def evaluate_model(dataset_name, detector):
    """
    Evaluate the model using COCO metrics (AP, IoU) on the KITTI MOTS dataset.
    
    Args:
        dataset_name (str): Dataset name.
        detector (Detector): The initialized Detector instance.
    """  
    print("Starting model evaluation...")

    # Create COCO evaluator
    evaluator = COCOEvaluator(dataset_name, output_dir="evaluation/")
    
    # Build detection test dataloader
    val_loader = build_detection_test_loader(detector.cfg, dataset_name)
    
    print("Running inference on dataset...")
    for batch in val_loader:
        img = batch[0]["file_name"]
        print(f"Processing image: {img}")

        # Run model inference
        outputs = detector.predictor(cv2.imread(img))

        if "instances" in outputs and len(outputs["instances"]) > 0:
            print(f"Detections found in image {img}:")
            print(outputs["instances"].to("cpu")) 
        else:
            print(f"⚠ No detections in image {img}")

    # Perform evaluation
    print(inference_on_dataset(detector.predictor.model, val_loader, evaluator))


if __name__ == '__main__':

    # Register dataset
    dataset_path = consts.KITTI_MOTS_PATH
    register_kitti_mots(dataset_path)

    dataset_name = "kitti_mots_training"
    # dataset_dicts = DatasetCatalog.get(dataset_name)

    # Initialize Detector
    detector = Detector()

    # Evaluate the model
    evaluate_model(dataset_name, detector)