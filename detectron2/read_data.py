from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import os
import numpy as np
from pycocotools.mask import decode

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
    annotations_dir = os.path.join(dataset_path, "instances_txt") 
    dataset_dicts = []
    
    kitti_to_coco = {2: 2, 1: 0}    # Car (2) -> COCO: Car (2), 
                                    # Pedestrian (1) -> COCO: Person (0)

    for folder in sorted(os.listdir(images_dir)):
        folder_path = os.path.join(images_dir, folder)
        annotation_file = os.path.join(annotations_dir, f"{folder}.txt")
        
        print(f"Processing folder: {folder}") 
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Missing annotation file for {folder}")
            continue
        
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
        
        frame_to_annotations = {}
        
        for line in annotations:
            parts = line.strip().split(" ")
            frame_id, _, class_id, h, w = map(int, parts[:5])
            rle_mask = " ".join(parts[5:])
            
            img_path = os.path.join(folder_path, f"{frame_id:06d}.png")
            if not os.path.exists(img_path):
                continue  # Skip missing images
            
            rle = {'size': [h, w], 'counts': rle_mask.encode('utf-8')}
            binary_mask = decode(rle)  # Convert RLE to binary mask
            
            bbox = mask_to_bbox(binary_mask)
            if bbox is None:
                continue
            
            if frame_id not in frame_to_annotations:
                frame_to_annotations[frame_id] = {
                    "file_name": img_path,
                    "image_id": len(dataset_dicts),
                    "height": h,
                    "width": w,
                    "annotations": []
                }
            
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [binary_mask.tolist()], 
                "category_id": kitti_to_coco.get(class_id, -1),
            }
            frame_to_annotations[frame_id]["annotations"].append(obj)
        
        dataset_dicts.extend(frame_to_annotations.values())
    
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
            thing_classes=["Car", "Pedestrian"],            # Match COCO class names
            thing_dataset_id_to_contiguous_id={0:0, 2:1},   # Person --> 0, Car --> 1
            evaluator_type="coco",
        )

        print(f"Registered {dataset_name} dataset.")