from datasets import Dataset
import numpy as np
import albumentations as A
from transformers import AutoImageProcessor

import albumentations as A


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples: Dataset,
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool=False
):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image, image_id, boxes, class_labels, area in zip(
        examples["image"], examples["image_id"], examples["boxes"], examples["class_labels"], examples["area"]
    ):
        # Image should be numpy array
        image = np.array(image.convert("RGB"))
        # apply augmentations
        output = transform(image=image, bboxes=boxes, category=class_labels)
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], area, output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def augment_and_transform_batch_deart(
    examples: Dataset,
    image_processor: AutoImageProcessor,
):
    
    images, annotations = [], []
    for image, img_id, img_annotations in zip(examples['image'], examples['image_id'], examples['annotations']):
        images.append(np.array(image.convert("RGB"))[:, :, ::-1])
        if len(img_annotations) > 0:
            annotations.append({
                "image_id": img_id,
                "annotations": img_annotations
            })
        else:
            annotations.append({
                "image_id": img_id,
                "annotations": []
            })
    return image_processor(images=images, annotations=annotations, return_tensors="pt")