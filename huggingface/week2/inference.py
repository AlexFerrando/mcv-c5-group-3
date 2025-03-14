import consts
import torch
import torch.nn as nn
import os

from consts import DetectionResults
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerModel, Mask2FormerImageProcessor
from typing import List, Tuple, Union
from read_data import VideoDataset
from tqdm import tqdm


def load_model(model_name: str = consts.MASK2FORMER) -> Tuple[Mask2FormerModel, Mask2FormerImageProcessor]:
    """
    Load model, processor, and determine device.
    
    Args:
        model_name: Name or path of the pre-trained model
    
    Returns:
        Tuple containing model, image processor, and device
    """

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerModel.from_pretrained(model_name)
    
    return model, image_processor



def run_instance_segmentation(
        model: Mask2FormerModel, 
        image_processor: Mask2FormerImageProcessor, 
        images: Union[Image.Image, List[Image.Image]], 
        device: torch.device
    ) -> List[DetectionResults]:
    """
    Run object detection inference on one or multiple images.
    
    Args:
        model: Object detection model
        image_processor: Image processor for the model
        images: Single image or list of images
        device: Device to run inference on
        threshold: Confidence threshold for detections
    
    Returns:
        List of detection results for each image
    """
    # Convert single image to list if needed
    if not isinstance(images, list):
        images = [images]
    
    inputs = image_processor(images=images, return_tensors="pt")

    # Forward pass
    model.to(device)
    with torch.no_grad():
        outputs = model.forward(**inputs.to(device))

    # Crear el tensor de target_sizes
    img_size = torch.tensor([images[0].size[1], images[0].size[0]], dtype=torch.float32)
    target_sizes = img_size.repeat(len(images), 1).to(device)

    # Perform post-processing to get instance segmentation map
    pred_instance_map = image_processor.post_process_instance_segmentation(
        outputs, target_sizes=target_sizes
    )[0]

    results = pred_instance_map
    # Filter the detection to get only the desired ones: 'car' (id=3 in DeTR) and 'person' (id=1 in DeTR)
    # Post process results to get apropiate format
    print(results)
            
    return results



if __name__ == '__main__':

    # DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'
    #DATASET_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS'
    DATASET_PATH = consts.KITTI_MOTS_PATH_ALEX

    # Get video names
    videos = os.listdir(os.path.join(DATASET_PATH, 'training/image_02'))
    videos = [video for video in videos if not video.startswith('.')]

    # Load model
    model, image_processor = load_model()
    data_loader = VideoDataset(DATASET_PATH)
    
    for video in tqdm(videos, desc="Processing videos", unit="video"):
        
        video_data = data_loader.load_video(video)
        frames = video_data['image']
        
        batch_size = 4
        for i in tqdm(range(0, len(frames), batch_size)):
            batch_frames = frames[i:i + batch_size]
            predictions = run_instance_segmentation(model, image_processor, batch_frames, device='cuda')

    print("Instance segmentation with Mask2Former off-the-shelf finished!")

