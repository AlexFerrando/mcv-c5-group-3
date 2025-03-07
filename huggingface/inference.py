import consts
import torch

from consts import DetectionResults
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from typing import List, Tuple, Optional, Union, Dict
from read_data import read_data
from consts import inverse_mapping_class_id
import json

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
                 threshold: float = 0.9,
                 output_format: str = 'coco') -> List[DetectionResults]:
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

    # Filter the detection to get only the desired ones: 'car' (id=3 in DeTR) and 'person' (id=1 in DeTR)
    filtered_detections = filter_detections_by_id(batch_results, id = [1,3])

    # Post process results to get apropiate format
    if output_format == 'coco':
        results = coco_reformat(results=filtered_detections, img_size=target_sizes)
    
    else:
        results = []
        for i, image in enumerate(images):
            result = filtered_detections[i]
            
            results.append(DetectionResults(
                scores=result['scores'],
                boxes=result['boxes'],
                labels=result['labels']
            ))
            
    return results


def filter_detections_by_id(results: List[Dict], id: List[int]=[1, 2]) -> List[Dict]:
    """
    Filter detection results to only keep detections with labels matching the specified IDs.
    
    Args:
        results: List of dictionaries, each with 'scores', 'labels', and 'boxes' keys
        id: List of label IDs to keep (defaults to [1, 2] which typically are 'car' and 'person')
    
    Returns:
        List of dictionaries with filtered detections
    """

    filtered_results = []
    
    for result in results:
        # Create a new dictionary to store filtered results
        filtered_result = {
            'scores': [],
            'labels': [],
            'boxes': []
        }
        
        # Only keep detections with matching labels
        for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
            # Check if this label is in our list of IDs to keep
            if label.item() in id:
                filtered_result['scores'].append(score)
                filtered_result['labels'].append(label)
                filtered_result['boxes'].append(box)
        
        # Convert lists back to tensors if needed
        if filtered_result['scores']:
            filtered_result['scores'] = torch.stack(filtered_result['scores'])
            filtered_result['labels'] = torch.stack(filtered_result['labels'])
            filtered_result['boxes'] = torch.stack(filtered_result['boxes'])
        
        filtered_results.append(filtered_result)
    
    return filtered_results


def coco_reformat(results: List[Dict], img_size: Tuple[int, int]) -> List[Dict]:
    """
    Converts a list of detection dictionaries to COCO prediction format.
    
    Args:
        results: List of dictionaries with 'scores', 'labels', and 'boxes' keys
        img_size: Tuple containing image dimensions
    
    Returns:
        List of dictionaries in COCO format with 'image_id', 'category_id', 'bbox', and 'score'
    """
    coco_predictions = []

    # Process each image's detection results
    for image_id, result in enumerate(results, 1):
        if 'scores' in result and len(result['scores']) > 0:
            for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
                # Convert box format from [x1, y1, x2, y2] to [x, y, width, height]
                box_data = box.tolist()
                x1, y1, x2, y2 = box_data
                width = x2 - x1
                height = y2 - y1
                
                # Create prediction entry
                prediction = {
                    'image_id': image_id,
                    'category_id': inverse_mapping_class_id('coco', int(label.item())),  # Map class IDs
                    'bbox': [x1, y1, width, height],
                    'score': float(score.item())  # Ensure it's a float
                }
                
                coco_predictions.append(prediction)

    return coco_predictions


if __name__ == '__main__':

    DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'

    # Load model
    model, image_processor, device = load_model()
    
    # Load dataset
    dataset = read_data(DATASET_PATH)
    dataset = dataset['train']['image']
    
    results = run_inference(model, image_processor, dataset, device)

    # Save the gt_coco as a JSON file
    output_json_path = 'results_coco_0000.json'
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
