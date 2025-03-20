from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
from typing import Tuple

class FoodDataset(Dataset):
    def __init__(self, data_path: str):
        data_path = Path(data_path)
        
        # Define df
        self.df = pd.read_csv(data_path / 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
        # Keep only the title and image name
        self.df = self.df[['Title', 'Image_Name']]
        # Remove rows with invalid 'Image_Name' entries (e.g., '#NAME?')
        self.df = self.df[self.df['Image_Name'] != '#NAME?']

        # Define image_path
        self.images_folder = data_path / 'Food Images/Food Images'
        
        print(f'Loaded {len(self.df)} samples')
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        row = self.df.iloc[idx]
        title = row['Title']
        # img_name we have to add .jpg
        img_name = row['Image_Name'] + '.jpg'
        img_path = self.images_folder / img_name
        
        return self.process_image(img_path), self.process_text(title)

    def process_image(self, img_path: Path) -> Image.Image:
        # to be implemented
        return Image.open(img_path).convert('RGB')
    
    def process_text(self, text: str):
        # to be implemented
        return text