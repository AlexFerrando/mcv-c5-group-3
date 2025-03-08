import torch
import numpy as np
import albumentations as A

from typing import Dict, Optional
from functools import partial
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import EvalPrediction, AutoImageProcessor, TrainingArguments, Trainer, TrainerCallback
from torchvision.transforms import functional as F

import torch
import torchvision.transforms.functional as F

from datasets import load_dataset
from inference import load_model
from read_data import read_data
import consts
from transformers.image_transforms import center_to_corners_format
import wandb
from finetuning_utils import augment_and_transform_batch, augment_and_transform_batch_deart


class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)



def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


# Define data collator
def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
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
    id2label: Optional[Dict[str, int]] = None
) -> Dict[str, float]:
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    for batch in targets:
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")

    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
    print(metrics)
    wandb.log(metrics)
    return metrics


# Load dataset
DATASET = 'DEART'

# Load model and image processor
model, image_processor, device = load_model(modified=True, for_dataset=DATASET)

# Define evaluation function
eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=model.config.id2label, threshold=0.5
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./outputs/alex/detr_finetuned",
    num_train_epochs=10,
    fp16=False,
    per_device_train_batch_size=8, # Change to 1 locally
    per_device_eval_batch_size=8, # Change to 1 locally
    dataloader_num_workers=4, # Change to 0 locally
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=False, # True if we want to load the best model at the end of training
    eval_strategy="steps",
    eval_steps=400,
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=False,
    logging_dir="./outputs/alex/logs",
    logging_steps=25,
)

if DATASET == 'KITTI':
    data = read_data(consts.KITTI_MOTS_PATH)
    unique_videos = sorted(data["train"].unique("video"))
    # Filter the dataset based on the sorted video split
    # Determine the split index (80% for training, 20% for testing)
    split_idx = int(0.8 * len(unique_videos))
    # Split the videos into training and testing sets
    train_videos = set(unique_videos[:split_idx])
    test_videos = set(unique_videos[split_idx:])
    train_data = data["train"].filter(lambda x: x["video"] in train_videos, num_proc=4)
    test_data = data["train"].filter(lambda x: x["video"] in test_videos, num_proc=4)
    # data = data["train"].train_test_split(test_size=0.2)
    # train_data = data["train"]
    # test_data = data["test"]

elif DATASET == 'DEART':
    data = load_dataset('davanstrien/deart')
    # Remove very large images (heigh or width > 2000)
    data['train'] = data['train'].filter(lambda x: x['width'] <= 2000 and x['height'] <= 2000, num_proc=4)
    data = data['train'].train_test_split(test_size=0.2)
    train_data = data['train']
    test_data = data['test']


if DATASET == 'KITTI':
    # Clean and transform data
    train_augment_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )

    test_augment_and_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )

    test_transform_batch = partial(
        augment_and_transform_batch, transform=test_augment_and_transform, image_processor=image_processor
    )
elif DATASET == 'DEART':
    train_transform_batch = partial(
        augment_and_transform_batch_deart, image_processor=image_processor
    )

    test_transform_batch = partial(
        augment_and_transform_batch_deart, image_processor=image_processor
    )

train_data = train_data.with_transform(train_transform_batch)
test_data = test_data.with_transform(test_transform_batch)

# Setup Wandb
wandb.login(key='395ee0b4fb2e10004d480c7d2ffe03b236345ddc')
wandb.init(
    project="c6-week1",
    name="detr_finetuning_20epochs_DEART",
    config=training_args.to_dict()  # Log training arguments
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=train_data,
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)
trainer.add_callback(WandbCallback())

# Start training
trainer.train()
wandb.finish()