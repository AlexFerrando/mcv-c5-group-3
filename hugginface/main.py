import consts
import torch
import os
from PIL import Image, ImageDraw
from read_data import read_data
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from typing import List, Dict, Tuple, Optional, Union


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
                 threshold: float = 0.9) -> List[Dict]:
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
        
        results = []
        for _ , image in enumerate(images):
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            result = image_processor.post_process_object_detection(
                outputs, 
                threshold=threshold, 
                target_sizes=target_sizes
            )[0]
            results.append(result)
            
    return results


def print_detection_results(results: Dict, model_config: Dict) -> None:
    """
    Print detection results.
    
    Args:
        results: Detection results for a single image
        model_config: Model configuration containing label mapping
    """
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model_config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )


def visualize_detections(image: Image.Image, 
                        results: Dict, 
                        model_config: Dict,
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
    
    for scores , label, box in zip(results["scores"], results["labels"], results["boxes"]):
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


def main():
    # Load model
    model, image_processor, device = load_model()
    
    # Load data
    dataset = read_data(consts.KITTI_MOTS_PATH_RELATIVE)
    image = dataset['test']['image'][0]
    
    # Run inference
    results = run_inference(model, image_processor, image, device)[0]
    
    # Print results
    print_detection_results(results, model.config)
    
    # Visualize and save
    output_path = "output.jpg"
    visualize_detections(image, results, model.config, output_path)
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    main()