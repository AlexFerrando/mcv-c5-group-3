import numpy as np
import torch
import albumentations as A

from typing import List, Tuple, Dict
from datasets import Dataset
from transformers import Mask2FormerImageProcessor, BatchFeature
from pycocotools.mask import decode
from numpy.typing import ArrayLike

def rle_to_instance_map(
    masks: List[str], class_labels: List[int], image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of RLE masks into separate instance and semantic segmentation maps.
    
    Args:
        masks (list): List of RLE-encoded masks (pycocotools format).
        class_labels (list): List of class labels corresponding to each mask.
        image_shape (tuple): Shape of the output segmentation map (height, width).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - A 2D instance segmentation map where each pixel is assigned a unique instance ID (1, 2, 3, ...).
              Background remains 0.
            - A 2D semantic segmentation map where each pixel is assigned its corresponding class label.
    """
    # Initialize empty instance and semantic segmentation maps
    instance_map = np.zeros(image_shape, dtype=np.uint16)  # Unique instance IDs
    semantic_map = np.zeros(image_shape, dtype=np.uint8)  # Class labels
    
    instance_id = 1  # Start from 1; 0 is reserved for background
    for rle_mask, class_label in zip(masks, class_labels):
        # Decode RLE into a binary mask
        rle = {'size': image_shape, 'counts': rle_mask.encode('utf-8')}
        binary_mask = decode(rle)
        
        # Assign the instance ID and class label to the corresponding pixels
        instance_map[binary_mask == 1] = instance_id
        semantic_map[binary_mask == 1] = class_label
        instance_id += 1  # Increment for the next instance
    
    return instance_map, semantic_map


def augment_and_transform_batch(
    examples: Dataset, transform: A.Compose, image_processor: Mask2FormerImageProcessor
) -> BatchFeature:
    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }
    
    for image, annotations, orig_size in zip(examples["image"], examples["annotations"], examples["orig_size"]):
        image_np = np.array(image.convert("RGB"))
        masks, class_labels = annotations["masks"], annotations["class_labels"]
        
        # Convert RLE masks to instance and semantic maps
        inst_map, sem_map = rle_to_instance_map(masks, class_labels, orig_size)
        
        # Stack semantic and instance maps into a 2-channel mask
        semantic_and_instance_masks = np.stack((sem_map, inst_map), axis=-1)
        
        # Apply augmentations
        output = transform(image=image_np, mask=semantic_and_instance_masks)
        aug_image = output["image"]
        aug_semantic_and_instance_masks = output["mask"]
        aug_instance_mask = aug_semantic_and_instance_masks[..., 1]
        
        # Create mapping from instance ID to semantic ID after augmentation
        unique_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
        instance_id_to_semantic_id = {
            instance_id: semantic_id for semantic_id, instance_id in unique_pairs
        }
        
        # Apply the image processor transformations: resizing, rescaling, normalization
        model_inputs = image_processor(
            images=[aug_image],
            segmentation_maps=[aug_instance_mask],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )
        
        batch["pixel_values"].append(model_inputs.pixel_values[0])
        batch["mask_labels"].append(model_inputs.mask_labels[0])
        batch["class_labels"].append(model_inputs.class_labels[0])
    
    return batch


def collate_fn(examples: BatchFeature) -> Dict[str, torch.Tensor]:
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    return batch