import consts
import torch
import os

from consts import DetectionResults
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from typing import List, Tuple, Optional, Union, Dict
from read_data_arnau import read_data, load_video
from consts import inverse_mapping_class_id
import json
from tqdm import tqdm


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


def save_predictions(predictions: List[Dict], video_name: str) -> None:
    
    # Definir la ruta de salida
    
    output_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/predictions'
    #output_folder = '/ghome/c5mcv03/mcv-c5-group-3/detectron2/evaluation/DeTR/off-the-shelf'

    os.makedirs(output_folder, exist_ok=True)  # Crea la carpeta si no existe

    # Definir la ruta del archivo
    output_json_path = f'{output_folder}/preds_coco_{video_name}.json'

    # Guardar el JSON
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)

if __name__ == '__main__':
    #DATASET_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS'
    DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'    
    
    videos = os.listdir(DATASET_PATH + '/training/image_02')
    
    model, image_processor, device = load_model()

    # Iterar sobre los videos
    for video in tqdm(videos, desc="Processing videos", unit="video"):
        if video == '.DS_Store':
            continue

        dataset = load_video(video)
        frames = dataset['image']
        
        # Dividir los frames en lotes de 10
        batch_size = 10
        all_predictions = []  # Aquí se almacenarán todas las predicciones para el video

        id_image_counter = 0

        # Añadir un tqdm para los lotes
        for i in tqdm(range(0, len(frames), batch_size), desc=f"Processing batches for {video}", unit="batch"):
            batch_frames = frames[i:i + batch_size]
            predictions = run_inference(model, image_processor, batch_frames, device)
            
            for i, prediction in enumerate(predictions):
                predictions[i]['image_id'] += id_image_counter
            
            all_predictions.extend(predictions)  # Agregar las predicciones del lote actual
            id_image_counter += batch_size

        # Guardar todas las predicciones del video en un solo archivo
        save_predictions(all_predictions, video_name=video)

    print("Inference for DeTR off-the-shelf finished!")
