import os
from typing import List, Dict
import numpy as np
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, Sequence
from pycocotools.mask import decode

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

def load_images_and_annotations(image_folder: str, annotation_folder: str) -> Dict:
    """Load image paths and corresponding annotation masks."""
    images, masks, track_ids, class_ids, frame_ids = [], [], [], [], []

    sequences = sorted(os.listdir(image_folder))
    for seq in sequences:
        seq_img_folder = os.path.join(image_folder, seq)
        seq_anno_file = os.path.join(annotation_folder, f"{seq}.txt")

        if not os.path.exists(seq_anno_file):
            continue  # Skip sequences with no annotations

        with open(seq_anno_file, "r") as f:
            annotations = f.readlines()

        frame_to_mask = {}
        for line in annotations:
            parts = line.strip().split(" ")
            frame_id, track_id, class_id, h, w = map(int, parts[:5])
            rle_mask = " ".join(parts[5:])

            img_path = os.path.join(seq_img_folder, f"{frame_id:06d}.png")
            if not os.path.exists(img_path):
                continue  # Skip missing images

            # Decode RLE mask
            rle = {'size': [h, w], 'counts': rle_mask.encode('utf-8')}
            mask = decode(rle).astype(np.uint8).tolist()  # Convert to a list of lists

            # Store annotations for this frame
            if frame_id not in frame_to_mask:
                frame_to_mask[frame_id] = {
                    "image": img_path,
                    "frame_id": frame_id,
                    "track_id": [],
                    "class_id": [],
                    "mask": [[0] * w for _ in range(h)]  # Initialize empty mask as a list
                }

            frame_to_mask[frame_id]["track_id"].append(track_id)
            frame_to_mask[frame_id]["class_id"].append(class_id)
            frame_to_mask[frame_id]["mask"] = np.maximum(frame_to_mask[frame_id]["mask"], mask).tolist()

        # Collect dataset entries
        for frame_data in frame_to_mask.values():
            images.append(frame_data["image"])
            masks.append(frame_data["mask"])
            track_ids.append(frame_data["track_id"])
            class_ids.append(frame_data["class_id"])
            frame_ids.append(frame_data["frame_id"])

    return {
        "image": images,
        "frame_id": frame_ids,
        "track_id": track_ids,
        "class_id": class_ids,
        "mask": masks
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
        "mask": Sequence(Sequence(Value("uint8")))  # FIX: Properly store 2D variable-size masks
    })

def read_data(data_path: str) -> DatasetDict:
    return DatasetDict({
        'train': read_train_data(data_path),
        'test': read_test_data(data_path)
    })