import os
import consts
import numpy as np
from typing import List, Optional
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, Sequence
from pycocotools.mask import decode


class VideoDataset:
    """Efficiently loads videos and annotations into Hugging Face's Dataset format."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @staticmethod
    def decode_rle_mask(rle_mask: str, img_h: int, img_w: int):
        """Decode RLE mask into a binary mask."""
        rle = {'size': [img_h, img_w], 'counts': rle_mask.encode('utf-8')}
        return np.array(decode(rle), dtype=np.uint8)

    def load_video_annotations(self, video_name: str, target_classes: Optional[List[int]] = None) -> dict:
        """
        Load and group annotations for a specific video.
        
        Returns a dictionary keyed by image_id, where each value contains:
        - video: video name
        - image: path to the image
        - image_id: unique image id
        - orig_size: [img_h, img_w]
        - annotations: {
                "class_labels": list of class labels,
                "masks": list of corresponding masks (as nested lists of float32)
            }
        """
        target_classes = target_classes or [1, 2]  # Default to cars & pedestrians
        annotation_file = os.path.join(self.dataset_path, f'instances_txt/{video_name}.txt')
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        frames_info = {}
        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) < 6:
                    print(f"Skipping invalid annotation: {line}")
                    continue

                try:
                    image_id, _, category_id, img_h, img_w = map(int, parts[:5])
                    rle_mask = " ".join(parts[5:])
                except ValueError:
                    print(f"Skipping corrupt annotation: {line}")
                    continue

                if category_id not in target_classes:
                    continue

                img_path = os.path.join(self.dataset_path, f'training/image_02/{video_name}/{image_id:06d}.png')

                if image_id not in frames_info:
                    frames_info[image_id] = {
                        "video": video_name,
                        "image": img_path,
                        "image_id": image_id,
                        "orig_size": [img_h, img_w],
                        "annotations": {
                            "class_labels": [],
                            "masks": []
                        }
                    }

                # Convert the mask to a nested list of float32 values
                frames_info[image_id]["annotations"]["class_labels"].append(
                    consts.KIITI_TO_MODEL_IDS[category_id]
                )
                frames_info[image_id]["annotations"]["masks"].append(rle_mask)

        return frames_info

    def load_video(self, video_name: str, target_classes: Optional[List[int]] = None) -> Dataset:
        """
        Load a specific video's annotations and return a Hugging Face Dataset.
        
        Each dataset entry will have the structure:
        {
            "video": video_name,
            "image": image_path,
            "image_id": image_id,
            "orig_size": [img_h, img_w],
            "annotations": {
                "class_labels": [list of class labels],
                "masks": [list of masks]
            }
        }
        """
        frames_info = self.load_video_annotations(video_name, target_classes)
        dataset = {
            "video": [],
            "image": [],
            "image_id": [],
            "orig_size": [],
            "annotations": []
        }
        for frame in frames_info.values():
            dataset["video"].append(frame["video"])
            dataset["image"].append(frame["image"])
            dataset["image_id"].append(frame["image_id"])
            dataset["orig_size"].append(frame["orig_size"])
            dataset["annotations"].append(frame["annotations"])

        return Dataset.from_dict(dataset, features=self.get_features())

    def load_data(self) -> Dataset:
        """Load all training videos into a single Dataset."""
        image_folder = os.path.join(self.dataset_path, 'training/image_02')
        dataset = {"video": [], "image": [], "image_id": [], "orig_size": [], "annotations": []}

        for video_name in sorted(os.listdir(image_folder)):
            if video_name.startswith('.'):
                continue

            try:
                video_data = self.load_video(video_name)

                # Extend lists instead of appending individual dicts
                for key in dataset:
                    dataset[key].extend(video_data[key])

            except FileNotFoundError:
                print(f"Warning: Missing annotations for {video_name}, skipping...")

        return Dataset.from_dict(dataset, features=self.get_features())

    @staticmethod
    def get_features() -> Features:
        """Define dataset features for Hugging Face `Dataset` with nested annotations."""
        return Features({
            "video": Value("string"),
            "image": HFImage(),
            "image_id": Value("int32"),
            "orig_size": Sequence(Value("int32")),
            "annotations": Features({
                "class_labels": Sequence(Value("int32")),
                "masks": Sequence(Value("string"))
            })
        })

    @staticmethod
    def split_data(data: Dataset) -> DatasetDict:
        """Split the dataset into training and test sets (80%, 20%)."""
        video_names = data.unique("video")
        video_names.sort()
        video_count = len(video_names)
        train_count = int(0.8 * video_count)

        train_videos = video_names[:train_count]
        test_videos = video_names[train_count:]

        train_data = data.filter(lambda x: x["video"] in train_videos)
        test_data = data.filter(lambda x: x["video"] in test_videos)

        return DatasetDict({
            "train": train_data,
            "test": test_data
        })