import torch
import numpy as np

from typing import Dict, Optional
from functools import partial
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import EvalPrediction, AutoImageProcessor
from torchvision.transforms import functional as F

import torch
import torchvision.transforms.functional as F

def collate_fn(batch):
    data = {}

    # Stack pixel values (images) from each sample in the batch
    data["pixel_values"] = torch.stack([F.to_tensor(sample["image"]) for sample in batch])

    # Prepare labels: stack each field from the "labels" dictionary
    data["labels"] = [{
        "boxes": torch.tensor(sample["bbox"], dtype=torch.float32),  # Shape: [num_boxes, 4]
        "class_labels": torch.tensor(sample["category_id"], dtype=torch.int64),  # Shape: [num_boxes]
        "area": torch.tensor(sample["area"], dtype=torch.float32),  # Shape: [num_boxes]
        "iscrowd": torch.tensor(sample["iscrowd"], dtype=torch.int64),  # Shape: [num_boxes]
        "orig_size": torch.tensor(sample["image"].size, dtype=torch.int32)  # Shape: [2]
    } for sample in batch]

    # Optionally handle "pixel_mask" if it exists in the sample
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([sample["pixel_mask"] for sample in batch])

    return data


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Dict[str, int]]=None
) -> Dict[str, float]:
    """
    Compute mean average mAP, mAR and their variants for the object detection task.
    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        image_processor (AutoImageProcessor): Image processor to post process model predictions.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)

        for image_target in batch:
            boxes = torch.tensor(image_target["bbox"])
            labels = torch.tensor(image_target["class_id"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")

    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
    return metrics


from inference import load_model

model, image_processor, device = load_model()

eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=model.config.id2label, threshold=0.5
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="detr_finetuned",
    num_train_epochs=30,
    fp16=False,
    per_device_train_batch_size=8,
    dataloader_num_workers=0,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=False,
)

from transformers import Trainer

from read_data import read_data
import consts

DATA_PATH = consts.KITTI_MOTS_PATH

data = read_data(DATA_PATH)
train_data = data["train"]
train_data = train_data.train_test_split(test_size=0.2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data["train"],
    eval_dataset=train_data["test"],
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)
trainer.train()
