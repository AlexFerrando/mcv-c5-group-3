import os
from PIL import Image
from typing import  List
from datasets import Dataset, DatasetDict

def load_videos(folder: str) -> List[Image.Image]:
    videos = [os.path.join(folder, f) for f in os.listdir(folder) if not f.startswith('.')]
    images = []
    for video in videos:
        frames = [os.path.join(video, f) for f in os.listdir(video) if not f.startswith('.')]
        frames.sort()
        for frame in frames:
            images.append(Image.open(frame))
            
    return images


def read_test_data(data_path: str) -> Dataset:
    video_folder = os.path.join(data_path, 'testing/image_02')
    return Dataset.from_dict({
        'image': load_videos(video_folder)
    })


def read_data(data_path: str) -> DatasetDict:
    return DatasetDict({
        # 'train': read_train_data(data_path),
        'test': read_test_data(data_path)
    })
    