from collections import defaultdict
import os
from typing import List, Dict, Optional
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, Sequence
from pycocotools.mask import toBbox

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


def decode_rle_mask(rle_mask: str, img_h: int, img_w: int):
    """Decode RLE mask into a bounding box (xcenter, ycenter, w, h)."""
    rle = {'size': [img_h, img_w], 'counts': rle_mask.encode('utf-8')}
    x1, y1, w, h = tuple(toBbox(rle))  # Ensure `toBbox` is properly imported
    return [x1, y1, x1 + w, y1 + h], h*w


def load_images_and_annotations_for_video(
    video_folder: str, 
    annotation_file: str, 
    target_classes: Optional[List[int]] = None
) -> Dict[int, Dict]:
    """
    Load annotations and convert them to COCO format, filtering for specific object classes.

    Args:
        video_folder (str): Path to the folder containing image frames.
        annotation_file (str): Path to the annotation file.
        target_classes (List[int], optional): List of class IDs to include (default: [1, 2] for 'car' and 'pedestrian').

    Returns:
        Dict[int, Dict]: Dictionary in COCO format with filtered annotations.
    """
    target_classes = target_classes or [1, 2]  # Default to cars & pedestrians if not provided
    frames_info = defaultdict(lambda: {
        "video": "",
        "image": "",
        "image_id": 0,
        "orig_size": [],
        "area": [],
        "track_id": [],
        "class_labels": [],
        "boxes": [],
    })

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    try:
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
    except Exception as e:
        raise RuntimeError(f"Error reading annotation file: {e}")

    for line in annotations:
        parts = line.strip().split(" ")
        if len(parts) < 6:
            print(f"Skipping invalid annotation line: {line}")  # Debugging
            continue

        try:
            image_id, track_id, category_id, img_h, img_w = map(int, parts[:5])
            rle_mask = " ".join(parts[5:])
        except ValueError:
            print(f"Skipping corrupt annotation line: {line}")  # Debugging
            continue

        if category_id not in target_classes:
            continue

        img_path = os.path.join(video_folder, f"{image_id:06d}.png")
        bbox, area = decode_rle_mask(rle_mask, img_h, img_w)

        frame_data = frames_info[image_id]
        frame_data["video"] = video_folder
        frame_data["image"] = img_path
        frame_data["image_id"] = image_id
        frame_data["orig_size"] = [img_h, img_w]
        frame_data["track_id"].append(track_id)
        frame_data["class_labels"].append(category_id)
        frame_data["boxes"].append(bbox)
        frame_data["area"].append(area)

    return dict(frames_info)




def load_video(video_name: str):
    """Load image paths and corresponding annotation masks."""

    #DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'
    DATASET_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS'

    image_folder = DATASET_PATH+f'/training/image_02/{video_name}'
    annotation_folder = DATASET_PATH+f'/instances_txt/{video_name}.txt'

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    if not os.path.exists(annotation_folder):
        raise FileNotFoundError(f"Annotation folder not found: {annotation_folder}")

    dataset = {
        "video": [], "image": [], "boxes": [], "track_id": [], "class_labels": [],
        "image_id": [], "orig_size": [], "area": [], "iscrowd": []
    }


    frame_to_mask = load_images_and_annotations_for_video(image_folder, annotation_folder)

    for frame_data in frame_to_mask.values():
        dataset["video"].append(frame_data["video"])
        dataset["image"].append(frame_data["image"])
        dataset["boxes"].append(frame_data["boxes"])
        dataset["track_id"].append(frame_data["track_id"])
        dataset["class_labels"].append(frame_data["class_labels"])
        dataset["image_id"].append(frame_data["image_id"])
        dataset["orig_size"].append(frame_data["orig_size"])
        dataset["area"].append(frame_data["area"])
        dataset["iscrowd"].append(frame_data["iscrowd"])

    video = Dataset.from_dict(dataset, features=get_train_features())

    return video
        

def load_images_and_annotations(image_folder: str, annotation_folder: str) -> Dict[str, List]:
    """Load image paths and corresponding annotation masks."""

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    if not os.path.exists(annotation_folder):
        raise FileNotFoundError(f"Annotation folder not found: {annotation_folder}")

    dataset = {
        "video": [], "image": [], "boxes": [], "track_id": [], "class_labels": [],
        "image_id": [], "orig_size": [], "area": [],
    }

    for seq in sorted(os.listdir(image_folder)):
        if seq.startswith('.'):
            continue
        
        seq_img_folder = os.path.join(image_folder, seq)
        seq_anno_file = os.path.join(annotation_folder, f"{seq}.txt")

        if not os.path.exists(seq_anno_file):
            print(f"Warning: Annotation file missing for sequence {seq}. Skipping...")
            continue

        frame_to_mask = load_images_and_annotations_for_video(seq_img_folder, seq_anno_file)

        for frame_data in frame_to_mask.values():
            dataset["video"].append(frame_data["video"])
            dataset["image"].append(frame_data["image"])
            dataset["boxes"].append(frame_data["boxes"])
            dataset["track_id"].append(frame_data["track_id"])
            dataset["class_labels"].append(frame_data["class_labels"])
            dataset["image_id"].append(frame_data["image_id"])
            dataset["orig_size"].append(frame_data["orig_size"])
            dataset["area"].append(frame_data["area"])

    return dataset


def read_test_data(data_path: str) -> Dataset:
    image_folder = os.path.join(data_path, 'testing/image_02')
    images = load_video_frames(image_folder)
    return Dataset.from_dict({'image': images}, features=Features({'image': HFImage()}))

def read_train_data(data_path: str) -> Dataset:
    image_folder = os.path.join(data_path, 'training/image_02')
    annotation_folder = os.path.join(data_path, 'instances_txt')
    data = load_images_and_annotations(image_folder, annotation_folder)
    return Dataset.from_dict(data, features=get_train_features())

def get_train_features() -> Features:
    """Define dataset features for Hugging Face `Dataset`."""
    return Features({
        "video": Value("string"),
        "image": HFImage(),  # Image path, automatically converted to PIL
        "image_id": Value("int32"),
        "track_id": Sequence(Value("int32")),
        "class_labels": Sequence(Value("int32")),
        "boxes": Sequence(Sequence(Value("float32"))),  # List of bounding boxes
        "orig_size": Sequence(Value("int32")),  # Alternative: Array(2, "int32")
        "area": Sequence(Value("float32")),  # Consistency with bbox
    })

def read_data(data_path: str) -> DatasetDict:
    return DatasetDict({
        'train': read_train_data(data_path),
        'test': read_test_data(data_path)
    })