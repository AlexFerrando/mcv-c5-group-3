from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import os
from PIL import Image

def load_kitti_mots_dataset(dataset_path, split="training"):
    """
    Loads KITTI MOTS dataset and converts it into Detectron2's standard format.

    Args:
        dataset_path (str): Path to KITTI MOTS dataset.
        split (str): "training" or "testing".

    Returns:
        list: A list of dictionaries, each representing an image and its annotations.
    """
    # Define the paths for images and annotation files
    images_dir = os.path.join(dataset_path, split, "image_02")
    annotations_dir = os.path.join(dataset_path, "instances_txt")

    dataset_dicts = []

    # Loop over all folders (e.g., '0000', '0001', ...) in the images_dir
    for folder in sorted(os.listdir(images_dir)):
        folder_path = os.path.join(images_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Process each image in the folder
        for frame_id, img_file in enumerate(sorted(os.listdir(folder_path))):
            if not img_file.endswith(".png"):
                continue

            image_path = os.path.join(folder_path, img_file)
            height, width = Image.open(image_path).size[::-1]

            record = {
                "file_name": image_path,
                "image_id": len(dataset_dicts),
                "height": height,
                "width": width,
                "annotations": [],
            }

            # Look for the correct annotation file
            annotation_file = os.path.join(annotations_dir, f"{folder}.txt")

            if not os.path.exists(annotation_file):
                continue

            # Read annotations for this frame
            with open(annotation_file, "r") as f:
                annotations = f.readlines()

            # Process each annotation for this frame
            for line in annotations:
                try:
                    fields = list(map(int, line.split()))
                except ValueError:
                    continue  # Skip malformed lines

                # Frame format: frame_id, _, category_id, _, _, x1, y1, x2, y2, ...
                frame, _, category_id, _, _, x1, y1, x2, y2, *_ = fields[:10]

                if frame != frame_id:
                    continue

                obj = {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": category_id - 1,  # KITTI MOTS classes start from 1
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
            thing_classes=["Person", "Bicycle", "Car"],
            thing_dataset_id_to_contiguous_id={1: 2, 2: 0},
            evaluator_type="coco",
        )

        print(f"Registered {dataset_name} dataset.")