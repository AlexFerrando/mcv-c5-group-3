from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import os
from PIL import Image
import numpy as np
import cv2

def mask_to_bbox(mask):
    """
    Converts a binary mask to a bounding box (x_min, y_min, width, height).
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        print("Warning: No bounding box detected from mask.") 
        return None 
    
    bbox = [int(np.min(x_indices)), int(np.min(y_indices)),
            int(np.max(x_indices) - np.min(x_indices)),
            int(np.max(y_indices) - np.min(y_indices))]
    
    print(f"Generated BBox: {bbox}") 
    return bbox

def load_kitti_mots_dataset(dataset_path, split="training"):
    images_dir = os.path.join(dataset_path, split, "image_02")
    instances_dir = os.path.join(dataset_path, "instances") 
    dataset_dicts = []
    
    kitti_to_coco = {2: 0, 1: 1}  # KITTI MOTS: Car (2) -> COCO: Car (0), 
                                                # Pedestrian (1) -> COCO: Person (1)

    for folder in sorted(os.listdir(images_dir)):
        folder_path = os.path.join(images_dir, folder)
        instance_folder_path = os.path.join(instances_dir, folder)
        
        print(f"Processing folder: {folder}") 
        
        for _, img_file in enumerate(sorted(os.listdir(folder_path))):
            if not img_file.endswith(".png"):
                continue

            image_path = os.path.join(folder_path, img_file)
            instance_path = os.path.join(instance_folder_path, img_file)
            
            height, width = Image.open(image_path).size[::-1]
            
            record = {
                "file_name": image_path,
                "image_id": len(dataset_dicts),
                "height": height,
                "width": width,
                "annotations": []
            }
            
            if not os.path.exists(instance_path):
                print(f"Warning: Instance mask missing for {img_file}")
                continue
            
            print(f"Processing image: {img_file}") 
            
            instance_mask = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
            if instance_mask is None:
                print(f"Error: Could not read instance mask {instance_path}")
                continue
            
            unique_instance_ids = np.unique(instance_mask)
            print(f"Unique instance IDs in mask: {unique_instance_ids}") 
            
            for instance_id in unique_instance_ids:
                # Background
                if instance_id == 0:
                    continue 
                
                category_id = instance_id // 1000
                if category_id not in kitti_to_coco:
                    continue
                
                binary_mask = (instance_mask == instance_id).astype(np.uint8)
                bbox = mask_to_bbox(binary_mask)
                if bbox is None:
                    continue
                
                print(f"Instance ID: {instance_id}, Category: {category_id}, BBox: {bbox}") 
                
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": kitti_to_coco[category_id],
                }
                record["annotations"].append(obj)
            
            dataset_dicts.append(record)
    
    return dataset_dicts

def register_kitti_mots(dataset_path):
    """
    Registers KITTI MOTS dataset with Detectron2.
    
    Args:
        dataset_path (str): Path to KITTI MOTS dataset.
    """
    for split in ["training", "testing"]:
        dataset_name = f"kitti_mots_{split}"

        # Register dataset
        DatasetCatalog.register(dataset_name, 
                                lambda d=split: load_kitti_mots_dataset(dataset_path, d))

        # Define metadata
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["Car", "Pedestrian"],  # KITTI MOTS classes
            evaluator_type="coco",
        )

        print(f"Registered {dataset_name} dataset.")