import os
from typing import List, Dict
import numpy as np
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, Sequence
from pycocotools.mask import toBbox
from pycocotools.coco import COCO

def load_video_frames(folder: str) -> List[str]:
    """Load sorted image paths from a video sequence folder."""
    videos = sorted(os.listdir(folder))
    # Remove the ones starting with .
    videos = [v for v in videos if not v.startswith('.')]
    all_frames = []
    for video in videos:
        video_path = os.path.join(folder, video)
        frames = sorted(
            [os.path.join(video_path, f) for f in os.listdir(video_path) if not f.startswith('.')]
        )
        all_frames.extend(frames)
    return all_frames


def load_images_and_annotations_for_video(video_folder: str, annotation_file: str, target_classes: List[int] = [1, 2]) -> Dict:
    """
    Load annotations and convert to COCO format, filtering for specific object classes.
    
    Args:
        video_folder: Path to the folder containing image frames
        annotation_file: Path to the annotation file
        target_classes: List of class IDs to include (default: [1, 2] for 'car' and 'pedestrian')
    
    Returns:
        Dictionary in COCO format with filtered annotations
    """
    with open(annotation_file, "r") as f:
        annotations = f.readlines()

    frame_to_mask = {}
    image_id_counter = 1
    
    for line in annotations:
        parts = line.strip().split(" ")
        frame_id, track_id, category_id, h, w = map(int, parts[:5])
        rle_mask = " ".join(parts[5:])

        # Skip annotations for classes we're not interested in
        # if category_id not in target_classes:
        #     continue
            
        img_path = os.path.join(video_folder, f"{frame_id:06d}.png")
        if not os.path.exists(img_path):
            continue  # Skip missing images

        # Decode RLE mask
        rle = {'size': [h, w], 'counts': rle_mask.encode('utf-8')}
        bbox = toBbox(rle).tolist()
        
        # Calculate area from bbox [x, y, width, height]
        area_value = bbox[2] * bbox[3]  # width * height

        # Create or update the image entry
        if frame_id not in frame_to_mask:
            frame_to_mask[frame_id] = {
                "image": img_path,
                "frame_id": frame_id,
                "area": [],
                "orig_size": [],
                "track_id": [],
                "category_id": [],
                "bbox": [],
                "iscrowd": []
            }
            # Add image metadata for COCO format
            frame_to_mask[frame_id]["image_id"] = image_id_counter
            image_id_counter += 1
            
        # Store frame annotation
        frame_to_mask[frame_id]["track_id"].append(track_id)
        frame_to_mask[frame_id]["category_id"].append(category_id)
        frame_to_mask[frame_id]["bbox"].append(bbox)
        frame_to_mask[frame_id]["area"].append(area_value)
        frame_to_mask[frame_id]["orig_size"].append([h, w])
        frame_to_mask[frame_id]["iscrowd"].append(0)

    return frame_to_mask
        


def load_images_and_annotations(image_folder: str, annotation_folder: str) -> Dict:
    """Load image paths and corresponding annotation masks."""
    images, bboxes, track_ids, category_ids, frame_ids, orig_sizes, areas, iscrowds = [], [], [], [], [], [], [], []

    sequences = sorted(os.listdir(image_folder))
    for seq in sequences:
        seq_img_folder = os.path.join(image_folder, seq)
        seq_anno_file = os.path.join(annotation_folder, f"{seq}.txt")

        if not os.path.exists(seq_anno_file):
            continue  # Skip sequences with no annotations

        # Load annotations
        frame_to_mask = load_images_and_annotations_for_video(seq_img_folder, seq_anno_file)

        # Collect dataset entries
        for frame_data in frame_to_mask.values():
            images.append(frame_data["image"])
            bboxes.append(frame_data["bbox"])
            track_ids.append(frame_data["track_id"])
            category_ids.append(frame_data["category_id"])
            frame_ids.append(frame_data["frame_id"])
            orig_sizes.append(frame_data["orig_size"])
            areas.append(frame_data["area"])
            iscrowds.append(frame_data["iscrowd"])

    return {
        "image": images,
        "bbox": bboxes,
        "track_id": track_ids,
        "category_id": category_ids,
        "frame_id": frame_ids,
        "orig_size": orig_sizes,
        "area": areas,
        "iscrowd": iscrowds
    }


def read_test_data(data_path: str) -> Dataset:
    image_folder = os.path.join(data_path, 'testing/image_02')
    images = load_video_frames(image_folder)
    return Dataset.from_dict({'image': images}, features=Features({'image': HFImage()}))

def read_train_data(data_path: str) -> Dataset:
    image_folder = os.path.join(data_path, 'training/image_02')
    annotation_folder = os.path.join(data_path, 'instances_txt')
    data = load_images_and_annotations(image_folder, annotation_folder)
    return Dataset.from_dict(data, features=get_features())

def get_features() -> Features:
    """Define dataset features for Hugging Face `Dataset`."""
    return Features({
        "image": HFImage(),  # Image path, automatically converted to PIL
        "frame_id": Value("int32"),
        "track_id": Sequence(Value("int32")),
        "category_id": Sequence(Value("int32")),  # Match compute_metrics
        "bbox": Sequence(Sequence(Value("float32"))),  # Ensure correct format
        "orig_size": Sequence(Sequence(Value("int32"))),  # Store original image size (h, w)
        "area": Sequence(Value("float32")),  # Area of bounding box
        "iscrowd": Sequence(Value("int32"))  # Required field for COCO
    })

def read_data(data_path: str) -> DatasetDict:
    return DatasetDict({
        'train': read_train_data(data_path),
        'test': read_test_data(data_path)
    })

def read_annotations(annotation_file: str) -> COCO:
    return COCO(annotation_file)