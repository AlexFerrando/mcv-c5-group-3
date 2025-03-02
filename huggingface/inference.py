import consts
import torch

from consts import DetectionResults
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from typing import List, Tuple, Optional, Union, Dict
from read_data import read_data


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
                 output_format: str = 'other') -> List[DetectionResults]:
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

    # Filter the detection to get only the desired ones: 'car' (id=1) and 'pedestrian' (id=2)
    filtered_detections = filter_and_correct_detections(batch_results, model.config.id2label, model.config.label2id)
    return filtered_detections


def filter_and_correct_detections(results: List[Dict], id2label: Dict, label2id: Dict) -> List[Dict]:
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
            if label.item() == label2id['car']:
                # Correct the label ID if needed
                if label.item() == label2id['car']:
                    label = torch.tensor(label2id['car'], device=label.device)
                filtered_result['scores'].append(score)
                filtered_result['labels'].append(label)
                filtered_result['boxes'].append(box)
        
        if len(filtered_result['scores']) == 0:
            continue
        filtered_result['scores'] = torch.stack(filtered_result['scores'])
        filtered_result['labels'] = torch.stack(filtered_result['labels'])
        filtered_result['boxes'] = torch.stack(filtered_result['boxes'])
        
        filtered_results.append(filtered_result)
    
    return filtered_results


def coco_reformat(results: List[Dict], img_size: Tuple[int, int]) -> Dict:
    """
    Converts a list of detection dictionaries to a single COCO format dictionary.
    
    Args:
        results: List of dictionaries with 'scores', 'labels', and 'boxes' keys
    
    Returns:
        Dictionary in COCO format with categories, images, and annotations
    """
    # Create the basic COCO format structure
    coco_dict = {
        'categories': [
            {'id': 1, 'name': 'car', 'supercategory': 'vehicle'},
            {'id': 2, 'name': 'pedestrian', 'supercategory': 'person'}
        ],
        'images': [],
        'annotations': []
    }
    
    annotation_id = 1  # Running annotation ID counter
    
    # Process each image's detection results
    for image_id, result in enumerate(results, 1):
        # Add image entry
        coco_dict['images'].append({
            'id': image_id,
            'file_name': f'image_{image_id}.jpg',  # Placeholder filename
            'width': int(img_size[0][0]),
            'height': int(img_size[0][1]),
        })
        
        # Process the detections for this image
        if 'scores' in result and len(result['scores']) > 0:
            for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
                # Convert box format from [x1, y1, x2, y2] to [x, y, width, height]
                box_data = box.tolist()
                x1, y1, x2, y2 = box_data
                width = x2 - x1
                height = y2 - y1
                
                # Create annotation entry
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': int(label.item()),
                    'bbox': [x1, y1, width, height],
                    'area': width * height,
                    'segmentation': [],
                    'iscrowd': 0,
                    'score': float(score.item())
                }
                
                coco_dict['annotations'].append(annotation)
                annotation_id += 1
    
    return coco_dict


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


if __name__ == '__main__':
    
    # Load model
    model, image_processor, device = load_model()
    
    # Load dataset
    dataset = read_data(consts.KITTI_MOTS_PATH_ALEX)
    dataset = dataset['train']['image'][0:10]
    
    # Run inference
    results = run_inference(model, image_processor, dataset, device)
    
    # Print results
    print_detection_results(results, model.config)
    
    # Visualize and save
    for i, image in enumerate(dataset):
        
        output_path = f"output_{i}.jpg"
        visualize_detections(image, results[i], model.config, output_path)
        print(f"Output saved to {output_path}")
