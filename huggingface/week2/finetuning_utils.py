import numpy as np
import torch

from typing import List, Tuple
from datasets import Dataset
from transformers import Mask2FormerImageProcessor, BatchFeature
from pycocotools.mask import decode
from numpy.typing import ArrayLike

def transform(examples: Dataset, image_processor: Mask2FormerImageProcessor) -> BatchFeature:
    """Apply image processor to the examples."""
    images = []
    segmentation_maps = []
    for image, annotations, orig_size in zip(examples["image"], examples["annotations"], examples["orig_size"]):
        images.append(np.array(image.convert("RGB")))
        masks, class_labels = annotations["masks"], annotations["class_labels"]
        
        # Convert RLE masks to segmentation map
        seg_map = rle_to_segmentation_map(masks, class_labels, orig_size)
        segmentation_maps.append(seg_map)
    
    result = image_processor.preprocess(images=images, segmentation_maps=segmentation_maps, return_tensors="pt")
    return result

def rle_to_segmentation_map(masks: List[str], class_labels: List[int], image_shape: Tuple[int, int]) -> ArrayLike:
    """
    Convert a list of RLE masks into a single 2D segmentation map.
    
    Args:
        masks (list): List of RLE-encoded masks (pycocotools format).
        class_labels (list): List of class labels corresponding to each mask.
        image_shape (tuple): Shape of the output segmentation map (height, width).
    
    Returns:
        np.ndarray: A 2D segmentation map of shape (height, width), where each pixel
                    is assigned the corresponding class label.
    """
    # Initialize an empty segmentation map
    segmentation_map = np.zeros(image_shape, dtype=np.uint8)
    
    # Iterate over each mask and its corresponding class label
    for rle_mask, class_label in zip(masks, class_labels):
        # Decode RLE into a binary mask
        rle = {'size': image_shape, 'counts': rle_mask.encode('utf-8')}
        binary_mask = decode(rle)
        
        # Assign the class label to the corresponding pixels
        segmentation_map[binary_mask == 1] = class_label
    
    return segmentation_map


def collate_fn(features: List[BatchFeature]) -> BatchFeature:
    """Collate a list of features into a single batch feature."""
    data = {}
    for key in features[0]:
        data[key] = torch.stack([feature[key] for feature in features])
    return data