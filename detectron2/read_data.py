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
        return None 
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def load_kitti_mots_dataset(dataset_path, split="training"):
    images_dir = os.path.join(dataset_path, split, "image_02")
    annotations_dir = os.path.join(dataset_path, "instances_txt") 

    dataset_dicts = []
    kitti_to_coco = {1:0, 2:1}

    for folder in sorted(os.listdir(images_dir)):
        if folder.startswith('.'):
            continue
        
        folder_path = os.path.join(images_dir, folder)
        annotation_file = os.path.join(annotations_dir, f"{folder}.txt")
        
        print(f"Processing folder: {folder}") 
        if not os.path.exists(annotation_file):
            continue
        
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
        
        frame_to_annotations = {}
        
        for line in annotations:
            parts = line.strip().split(" ")
            frame_id, _, class_id, h, w = map(int, parts[:5])
            rle_mask = " ".join(parts[5:])
            img_path = os.path.join(folder_path, f"{frame_id:06d}.png")

            if not os.path.exists(img_path) or class_id not in kitti_to_coco:
                continue 
            
            rle = {'size': [h, w], 'counts': rle_mask.encode('utf-8')}
            binary_mask = decode(rle)  
            
            bbox = mask_to_bbox(binary_mask)
            if bbox is None:
                continue
            
            if frame_id not in frame_to_annotations:
                frame_to_annotations[frame_id] = {
                    "file_name": img_path,
                    "image_id": len(dataset_dicts) + len(frame_to_annotations),
                    "height": h,
                    "width": w,
                    "annotations": []
                }
            
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": kitti_to_coco[class_id],
                "segmentation": rle,
            }

            frame_to_annotations[frame_id]["annotations"].append(obj)
        
        dataset_dicts.extend(frame_to_annotations.values())
    
    return dataset_dicts

def register_kitti_mots(dataset_path, split="training"):
    """
    Registers KITTI MOTS dataset with Detectron2.
    
    Args:
        dataset_path (str): Path to KITTI MOTS dataset.
    """
    dataset_name = f"kitti_mots_{split}"

    if dataset_name in DatasetCatalog.list():
        print(f"Dataset '{dataset_name}' is already registered.")
        return
    
    # Register dataset
    DatasetCatalog.register(dataset_name, lambda: load_kitti_mots_dataset(dataset_path, split))

    # Define metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["Person", "Car"],                # Match COCO class names
        thing_dataset_id_to_contiguous_id={1: 0, 2: 1}, # Person --> 0, Car --> 1
        evaluator_type="coco",
    )

    print(f"Successfully registered dataset: {dataset_name}")
