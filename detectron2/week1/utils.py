from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

from sklearn.model_selection import train_test_split
import json
import os
import random

from read_data import load_kitti_mots_dataset

def split_dataset(dataset_path):
    """Splits the KITTI MOTS dataset into 80% training and 20% validation."""
    
    # Load full dataset
    full_data = load_kitti_mots_dataset(dataset_path, "training")

    # Shuffle and split
    random.shuffle(full_data)
    train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)

    # Save JSON files for Detectron2 compatibility
    os.makedirs("splits", exist_ok=True)
    
    train_json_path = "splits/kitti_mots_train.json"
    val_json_path = "splits/kitti_mots_val.json"

    with open(train_json_path, "w") as f:
        json.dump(train_data, f)
    with open(val_json_path, "w") as f:
        json.dump(val_data, f)
        
    return train_json_path, val_json_path

def register_datasets(dataset_path):
    """Registers KITTI MOTS datasets with Detectron2 after splitting."""

    # Split dataset into train and validation
    train_json, val_json = split_dataset(dataset_path)

    dataset_train = "kitti_mots_train"
    dataset_val = "kitti_mots_val"

    # Register Train Dataset
    if dataset_train not in DatasetCatalog.list():
        register_coco_instances(dataset_train, {}, train_json, dataset_path)
        MetadataCatalog.get(dataset_train).set(thing_classes=["Car", "Pedestrian"])
        print(f"Registered dataset: {dataset_train}")

    # Register Validation Dataset
    if dataset_val not in DatasetCatalog.list():
        register_coco_instances(dataset_val, {}, val_json, dataset_path)
        MetadataCatalog.get(dataset_val).set(thing_classes=["Car", "Pedestrian"])
        print(f"Registered dataset: {dataset_val}")

