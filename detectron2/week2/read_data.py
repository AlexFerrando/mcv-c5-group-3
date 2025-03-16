from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import os
from PIL import Image
from pycocotools.mask import decode, toBbox
import pycocotools.mask as mask_utils

def load_kitti_mots_dataset(dataset_path, split="training"):
    images_dir = os.path.join(dataset_path, split, "image_02")
    annotations_dir = os.path.join(dataset_path, "instances_txt") 

    dataset_dicts = []

    class_map = {1: 2, 2: 0}
    
    sample_idx = indices[idx]
    sample = samples[sample_idx]
    
    sequence_id = sample['sequence_id']
    frame_id = sample['frame_id']
    image_path = sample['image_path']
    
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Create a unique image ID
    image_id = f"{sequence_id}_{frame_id}"
    
    # Look for the annotation file
    ann_file = f"{sequence_id}.txt"
    ann_path = os.path.join(instances_dir, ann_file)
    
    annotations = []
    if os.path.exists(ann_path):
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        
        # Filter annotations for this frame
        frame_annotations = [line for line in lines if int(line.strip().split()[0]) == int(frame_id)]
        
        for line in frame_annotations:
            parts = line.strip().split()
            if len(parts) >= 5:

                instance_full_id = int(parts[1])
                
                class_id = instance_full_id // 1000
                
                if class_id not in class_map:
                    continue
                
                # Extract RLE from the last part
                rle_str = parts[5]
                h, w = int(parts[3]), int(parts[4])
                rle_obj = {'counts': rle_str, 'size': [h, w]}
                
                # Convert RLE to bbox
                bbox = [int(coord) for coord in mask_utils.toBbox(rle_obj)]
                x, y, w, h = bbox
                bbox_xyxy = [x, y, x + w, y + h]
                
                segmentation = {
                    "size": [img_height, img_width],
                    "counts": rle_str.encode('utf-8') if isinstance(rle_str, str) else rle_str
                }
                
                # Create annotation dictionary
                annotation = {
                    "bbox": bbox_xyxy,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": segmentation,
                    "category_id": class_map[class_id],
                }
                annotations.append(annotation)
    
    record = {
        "file_name": image_path,
        "height": img_height,
        "width": img_width,
        "image_id": image_id,
        "annotations": annotations,
    }
    return record

def register_kitti_mots(dataset_path):
    """
    Registers KITTI MOTS dataset with Detectron2.
    
    Args:
        dataset_path (str): Path to KITTI MOTS dataset.
    """
    for split in ["training", "testing"]:
        dataset_name = f"kitti_mots_{split}"
        if dataset_name in DatasetCatalog.list():
            print(f"Removing existing dataset '{dataset_name}' to avoid cache issues...")
            DatasetCatalog.remove(dataset_name)
    
        # Register dataset
        DatasetCatalog.register(dataset_name, 
                                lambda d=split: load_kitti_mots_dataset(dataset_path, d))

        # Inference
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["Person", "Bicycle", "Car"],
            thing_dataset_id_to_contiguous_id={1: 2, 2: 0},
            evaluator_type="coco",
        )

        # Evaluation
        # MetadataCatalog.get(dataset_name).set(
        #     thing_classes=["Person", "Car"],
        #     thing_dataset_id_to_contiguous_id={1: 0, 2: 1},
        #     evaluator_type="coco",
        # )

        print(f"Registered {dataset_name} dataset.")
