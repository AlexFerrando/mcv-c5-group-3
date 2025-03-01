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
    images_dir = os.path.join(dataset_path, split, "image_02", "0000")
    annotations_file = os.path.join(dataset_path, "instances_txt", "0000.txt")

    dataset_dicts = []

    # Read annotations
    with open(annotations_file, "r") as f:
        annotations = f.readlines()

    # Process each image
    for frame_id, img_file in enumerate(sorted(os.listdir(images_dir))):
        if not img_file.endswith(".png"):
            continue

        image_path = os.path.join(images_dir, img_file)
        height, width = Image.open(image_path).size[::-1]

        record = {
            "file_name": image_path,
            "image_id": frame_id,
            "height": height,
            "width": width,
            "annotations": [],
        }

        # Read annotations for this frame
        for line in annotations:
            fields = list(map(int, line.split()))
            frame, _, category_id, _, _, x1, y1, x2, y2, _ = fields[:10]

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
            thing_classes=["Car", "Pedestrian"],  # KITTI MOTS classes
            evaluator_type="coco",
        )

        print(f"Registered {dataset_name} dataset.")