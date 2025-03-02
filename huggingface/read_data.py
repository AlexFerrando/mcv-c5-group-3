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


def load_images_and_annotations_for_video(video_folder: str, annotation_file: str) -> Dict:

    with open(annotation_file, "r") as f:
        annotations = f.readlines()

    frame_to_mask = {}
    for line in annotations:
        parts = line.strip().split(" ")
        frame_id, track_id, class_id, h, w = map(int, parts[:5])
        rle_mask = " ".join(parts[5:])

        img_path = os.path.join(video_folder, f"{frame_id:06d}.png")
        if not os.path.exists(img_path):
            continue  # Skip missing images


        # Decode RLE mask
        rle = {'size': [h, w], 'counts': rle_mask.encode('utf-8')}
        bbox = toBbox(rle).tolist()

        # Store annotations for this frame
        if frame_id not in frame_to_mask:
            frame_to_mask[frame_id] = {
                "image": img_path,
                "frame_id": frame_id,
                "track_id": [],
                "class_id": [],
                "bbox": []
            }

        frame_to_mask[frame_id]["track_id"].append(track_id)
        frame_to_mask[frame_id]["class_id"].append(class_id)
        frame_to_mask[frame_id]["bbox"].append(bbox)
    
    return frame_to_mask


def load_images_and_annotations(image_folder: str, annotation_folder: str) -> Dict:
    """Load image paths and corresponding annotation masks."""
    images, bboxes, track_ids, class_ids, frame_ids = [], [], [], [], []

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
            class_ids.append(frame_data["class_id"])
            frame_ids.append(frame_data["frame_id"])

    return {
        "image": images,
        "frame_id": frame_ids,
        "track_id": track_ids,
        "class_id": class_ids,
        "bbox": bboxes
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
        "track_id": Sequence(Value("int32")),  # List of integers
        "class_id": Sequence(Value("int32")),  # List of integers
        "bbox": Sequence(Sequence(Value("float32")))  # List of floats
    })

def read_data(data_path: str) -> DatasetDict:
    return DatasetDict({
        'train': read_train_data(data_path),
        'test': read_test_data(data_path)
    })

def read_annotations(annotation_file: str) -> COCO:
    return COCO(annotation_file)