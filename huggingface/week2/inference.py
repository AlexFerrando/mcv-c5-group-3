import consts
import torch
import torch.nn as nn
import os

from consts import DetectionResults
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Mask2FormerModel
from typing import List, Tuple, Optional, Union, Dict
from read_data import read_data, load_video
from consts import inverse_mapping_class_id
import json
from tqdm import tqdm


def load_model(model_name: str = consts.MODEL_NAME, modified: bool = False) -> Tuple[torch.nn.Module, AutoImageProcessor]:
    """
    Load model, processor, and determine device.
    
    Args:
        model_name: Name or path of the pre-trained model
    
    Returns:
        Tuple containing model, image processor, and device
    """

    image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
    model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-coco-instance")
    
    return model, image_processor



def run_instance_segmentation(
        model: torch.nn.Module, 
        image_processor: AutoImageProcessor, 
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
    with torch.no_grad():
        outputs = model(**inputs.to(device))

    # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

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
            
    return results



if __name__ == '__main__':

    #DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'
    DATASET_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS'

    # Get video names
    videos = os.listdir(DATASET_PATH+'/training/image_02')

    # Load model
    model, image_processor, device = load_model()
    
    for video in tqdm(videos, desc="Processing videos", unit="video"):
        
        try:
            dataset = load_video(video)
        except:
            continue
        
        frames = dataset['image']
        
        predictions = run_instance_segmentation(model, image_processor, frames, device)

    print("Instance segmentation with Mask2Former off-the-shelf finished!")

