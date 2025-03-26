from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
from typing import Tuple
from transformers import ViTImageProcessor, PreTrainedTokenizer
import torch
import albumentations as A
import numpy as np

class FoodDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            feature_extractor: ViTImageProcessor,
            transform: A.Compose = A.Compose([A.NoOp()])
        ):
        data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        
        # Define df
        self.df = pd.read_csv(data_path / 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
        # Keep only the title and image name
        self.df = self.df[['Title', 'Image_Name']]
        # Remove rows with invalid 'Image_Name' entries (e.g., '#NAME?')
        self.df = self.df[self.df['Image_Name'] != '#NAME?']
        # Remove nans
        self.df = self.df.dropna() # There are 5 nans xd

        # TODO: Maybe remove pictures that were not food related?

        # Define image_path
        self.images_folder = data_path / 'Food Images/Food Images'
        
        print(f'Loaded {len(self.df)} samples')
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        title = row['Title']
        # img_name we have to add .jpg
        img_name = row['Image_Name'] + '.jpg'
        img_path = self.images_folder / img_name
        
        return self.process_image(img_path), self.process_text(title)

    def process_image(self, img_path: Path) -> torch.Tensor:
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image=np.array(image))['image']
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        return pixel_values.squeeze(0)
    
    def process_text(self, text: str) -> torch.Tensor:
        return torch.Tensor(
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.tokenizer.model_max_length + 1,
                truncation=True,
                padding="max_length"
            )
        )