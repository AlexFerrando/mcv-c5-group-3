import numpy as np
import torch
import albumentations as A
import wandb

from dataclasses import dataclass
from typing import List, Tuple, Dict, Mapping
from datasets import Dataset
from transformers import Mask2FormerImageProcessor, BatchFeature, AutoImageProcessor, TrainerCallback
from pycocotools.mask import decode
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.trainer import EvalPrediction

class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)

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


# Code Evaluator extracted from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/instance-segmentation/run_instance_segmentation.py
@dataclass
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor

def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors

class Evaluator:
    """
    Compute metrics for the instance segmentation task.
    """

    def __init__(
        self,
        image_processor: AutoImageProcessor,
        id2label: Mapping[int, str],
        threshold: float = 0.0,
    ):
        """
        Initialize evaluator with image processor, id2label mapping and threshold for filtering predictions.

        Args:
            image_processor (AutoImageProcessor): Image processor for
                `post_process_instance_segmentation` method.
            id2label (Mapping[int, str]): Mapping from class id to class name.
            threshold (float): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        """
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = self.get_metric()

    def get_metric(self):
        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        return metric

    def reset_metric(self):
        self.metric.reset()

    def postprocess_target_batch(self, target_batch) -> List[Dict[str, torch.Tensor]]:
        """Collect targets in a form of list of dictionaries with keys "masks", "labels"."""
        batch_masks = target_batch[0]
        batch_labels = target_batch[1]
        post_processed_targets = []
        for masks, labels in zip(batch_masks, batch_labels):
            post_processed_targets.append(
                {
                    "masks": masks.to(dtype=torch.bool),
                    "labels": labels,
                }
            )
        return post_processed_targets

    def get_target_sizes(self, post_processed_targets) -> List[List[int]]:
        target_sizes = []
        for target in post_processed_targets:
            target_sizes.append(target["masks"].shape[-2:])
        return target_sizes

    def postprocess_prediction_batch(self, prediction_batch, target_sizes) -> List[Dict[str, torch.Tensor]]:
        """Collect predictions in a form of list of dictionaries with keys "masks", "labels", "scores"."""

        model_output = ModelOutput(class_queries_logits=prediction_batch[0], masks_queries_logits=prediction_batch[1])
        post_processed_output = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        post_processed_predictions = []
        for image_predictions, target_size in zip(post_processed_output, target_sizes):
            if image_predictions["segments_info"]:
                post_processed_image_prediction = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]]),
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]]),
                }
            else:
                # for void predictions, we need to provide empty tensors
                post_processed_image_prediction = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                    "labels": torch.tensor([]),
                    "scores": torch.tensor([]),
                }
            post_processed_predictions.append(post_processed_image_prediction)

        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, compute_result: bool = False) -> Mapping[str, float]:
        """
        Update metrics with current evaluation results and return metrics if `compute_result` is True.

        Args:
            evaluation_results (EvalPrediction): Predictions and targets from evaluation.
            compute_result (bool): Whether to compute and return metrics.

        Returns:
            Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
        """
        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)

        # For metric computation we need to provide:
        #  - targets in a form of list of dictionaries with keys "masks", "labels"
        #  - predictions in a form of list of dictionaries with keys "masks", "labels", "scores"
        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(prediction_batch, target_sizes)

        # Compute metrics
        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return

        metrics = self.metric.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        print(metrics)
        wandb.log(metrics)

        # Reset metric for next evaluation
        self.reset_metric()

        return metrics