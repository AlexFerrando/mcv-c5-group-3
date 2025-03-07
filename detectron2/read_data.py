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
    kitti_to_coco = {1: 0, 2: 1} # Mapping KITTI to COCO

    for sequence_id  in sorted(os.listdir(images_dir)):
        if sequence_id .startswith('.'):
            continue
        
        sequence_path = os.path.join(images_dir, sequence_id)
        annotation_file = os.path.join(annotations_dir, f"{sequence_id}.txt")
        
        print(f"Processing folder: {sequence_id}")
        if not os.path.exists(annotation_file):
            continue
        
        # Load annotations
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
        
        frame_to_annotations = {}
        
        for line in annotations:
            parts = line.strip().split()
            frame_id, instance_id, class_id, h, w = map(int, parts[:5])
            
            if class_id == 10:
                continue

            img_path = os.path.join(sequence_path, f"{frame_id:06d}.png")
            if not os.path.exists(img_path):
                continue 

            rle_mask = parts[5]
            rle = {'size': [h, w], 'counts': rle_mask}
            
            bbox = mask_utils.toBbox(rle).tolist()  # Convert RLE to bbox
            if bbox is None:
                continue

            ann = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": kitti_to_coco.get(class_id, -1),
            }

            if frame_id not in frame_to_annotations:
                img = Image.open(img_path)
                img_width, img_height = img.size

                frame_to_annotations[frame_id] = {
                    "file_name": img_path,
                    "image_id": f"{sequence_id}_{frame_id}",
                    "height": img_height,
                    "width": img_width,
                    "annotations": []
                }

            frame_to_annotations[frame_id]["annotations"].append(ann)

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
