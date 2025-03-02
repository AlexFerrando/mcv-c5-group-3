import consts
import torch
import os
import json
import os

from consts import DetectionResults
from PIL import Image, ImageDraw
from read_data import read_data, load_video_frames, load_images_and_annotations_for_video
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from typing import List, Dict, Tuple, Optional, Union
from read_data import read_annotations
from transformers import DetrForObjectDetection


def load_model(model_name: str = consts.MODEL_NAME) -> Tuple[torch.nn.Module, AutoImageProcessor, torch.device]:
    """
    Load model, processor, and determine device.
    
    Args:
        model_name: Name or path of the pre-trained model
    
    Returns:
        Tuple containing model, image processor, and device
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.to(device)
    
    return model, image_processor, device


def run_inference(model: torch.nn.Module, 
                 image_processor: AutoImageProcessor, 
                 images: Union[Image.Image, List[Image.Image]], 
                 device: torch.device,
                 threshold: float = 0.9) -> List[DetectionResults]:
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
    
    with torch.no_grad():
        inputs = image_processor(images=images, return_tensors="pt")
        outputs = model(**inputs.to(device))

    # Crear el tensor de target_sizes
    img_size = torch.tensor([images[0].size[1], images[0].size[0]], dtype=torch.float32)
    target_sizes = img_size.repeat(len(images), 1).to(device)

    # Aply detecction to the hole batch
    batch_results = image_processor.post_process_object_detection(
        outputs, 
        threshold=threshold, 
        target_sizes=target_sizes
    )

    # Post process results to get apropiate format
    results = []
    for i, image in enumerate(images):
        result = batch_results[i]
        
        results.append(DetectionResults(
            scores=result['scores'],
            boxes=result['boxes'],
            labels=result['labels']
        ))
            
    return results


def print_detection_results(results: DetectionResults, model_config: torch.nn.Module) -> None:
    """
    Print detection results.
    
    Args:
        results: Detection results for a single image
        model_config: Model configuration containing label mapping
    """
    for i, _ in enumerate(results):
        for score, label, box in zip(results[i].scores, results[i].labels, results[i].boxes):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model_config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )


def visualize_detections(image: Image.Image, 
                        results: DetectionResults, 
                        model_config: torch.nn.Module,
                        output_path: Optional[str] = None) -> Image.Image:
    """
    Visualize object detection results on an image.
    
    Args:
        image: Original image
        results: Detection results
        model_config: Model configuration containing label mapping
        output_path: Path to save the output image (optional)
    
    Returns:
        Image with detections visualized
    """

    # Create a copy to avoid modifying the original
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    
    for label, box in zip(results.labels, results.boxes):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model_config.id2label[label.item()], fill="white")
    
    # Save the image if output path is provided
    if output_path:
        output_image.save(output_path)
    
    return output_image


def process_batch(model: torch.nn.Module,
                 image_processor: AutoImageProcessor,
                 images: List[Image.Image],
                 device: torch.device,
                 threshold: float = 0.9,
                 output_dir: Optional[str] = None) -> List[Dict]:
    """
    Process a batch of images for object detection.
    
    Args:
        model: Object detection model
        image_processor: Image processor for the model
        images: List of images to process
        device: Device to run inference on
        threshold: Confidence threshold for detections
        output_dir: Directory to save output visualizations (optional)
    
    Returns:
        List of detection results
    """
    results = run_inference(model, image_processor, images, device, threshold)
    
    for i, (image, result) in enumerate(zip(images, results)):
        print(f"Image {i}:")
        print_detection_results(result, model.config)
        
        if output_dir:
            output_path = os.path.join(output_dir, f"detection_{i}.jpg")
            visualize_detections(image, result, model.config, output_path)
    
    return results


def kitti_to_coco(kitti_file, output_json):
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "pedestrian", "supercategory": "person"},
        {"id": 2, "name": "car", "supercategory": "vehicle"}
    ]
    
    with open(kitti_file, "r") as f:
        lines = f.readlines()
    
    image_ids = {}
    ann_id = 1
    
    for line in lines:
        parts = line.strip().split(" ")
        
        frame_id = int(parts[0])
        track_id = int(parts[1])
        obj_class = int(parts[2])
        img_width = int(parts[3])
        img_height = int(parts[4])
        encoded_mask = parts[5]

        # Si la imagen aún no se ha agregado
        if frame_id not in image_ids:
            image_ids[frame_id] = len(image_ids) + 1
            images.append({
                "id": image_ids[frame_id],
                "file_name": f"{frame_id:06d}.png",
                "width": img_width,
                "height": img_height
            })

        annotations.append({
            "id": ann_id,
            "image_id": image_ids[frame_id],
            "category_id": obj_class,
            "segmentation": {
                "counts": encoded_mask,
                "size": [img_height, img_width]
            },
            "area": 0,  # Se puede calcular si se decodifica la máscara
            "bbox": [0, 0, 0, 0],  # Se puede calcular si se decodifica la máscara
            "iscrowd": 1
        })
        
        ann_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)



def main():

    video0000_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/training/image_02/0000'
    annotations_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/instances_txt/0000.txt'


    kitti_to_coco(annotations_folder, "coco_annotations.json")

    gt = read_annotations("coco_annotations.json")

    # Load model
    model, image_processor, device = load_model()
    
    # Load dataset
    dataset = read_data(consts.KITTI_MOTS_PATH_RELATIVE)
    dataset = dataset['train']['image'][0:10]
    
    # Run inference
    results = run_inference(model, image_processor, dataset, device)
    
    # Print results
    print_detection_results(results, model.config)
    
    # # Visualize and save
    # for i, image in enumerate(dataset):
        
    #     output_path = f"output_{i}.jpg"
    #     visualize_detections(image, results[i], model.config, output_path)
    #     print(f"Output saved to {output_path}")


if __name__ == '__main__':
    main()