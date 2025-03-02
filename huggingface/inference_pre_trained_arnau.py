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


from typing import List, Dict, Tuple

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
                        coco_results: Dict, 
                        image_id: int = 1,
                        output_path: Optional[str] = None) -> Image.Image:
    """
    Visualize object detection results on an image using COCO format data.
    
    Args:
        image: Original image
        coco_results: Detection results in COCO format
        image_id: ID of the image to visualize in the COCO results (default: 1)
        output_path: Path to save the output image (optional)
    
    Returns:
        Image with detections visualized
    """
    # Create a copy to avoid modifying the original
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    
    # Create a category ID to name mapping
    category_mapping = {cat['id']: cat['name'] for cat in coco_results['categories']}
    
    # Find annotations for the specified image ID
    annotations = [ann for ann in coco_results['annotations'] if ann['image_id'] == image_id]
    
    # Draw each detection
    for ann in annotations:
        # Extract bounding box in [x, y, width, height] format
        x, y, width, height = ann['bbox']
        x2, y2 = x + width, y + height
        
        # Get category name
        category_id = ann['category_id']
        category_name = category_mapping.get(category_id, f"Unknown-{category_id}")
        
        # Get confidence score if available
        score_text = f" {ann['score']:.2f}" if 'score' in ann else ""
        
        # Draw rectangle and label
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), f"{category_name}{score_text}", fill="white")
    
    # Save the image if output path is provided
    if output_path:
        output_image.save(output_path)
    
    return output_image


if __name__ == '__main__':

    # Load model
    model, image_processor, device = load_model()
    
    # Load dataset
    dataset = read_data(consts.KITTI_MOTS_PATH_RELATIVE)
    dataset = dataset['train']['image'][0:30]
    
    # Run inference
    results = run_inference(model, image_processor, dataset, device)

    # Guardar en un archivo JSON
    with open("predictions_30.json", "w") as file:
        json.dump(results, file, indent=4)  # `indent=4` para que el JSON sea legible

