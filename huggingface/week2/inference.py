import consts
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from consts import DetectionResults
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from typing import List, Tuple, Union, Dict
from read_data import VideoDataset
from tqdm import tqdm
import cv2

import json
from pycocotools import mask as mask_util


CONFIG = Mask2FormerConfig.from_pretrained(consts.MASK2FORMER)

def load_model(model_name: str = consts.MASK2FORMER) -> Tuple[Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor]:
    """
    Load model, processor, and determine device.
    
    Args:
        model_name: Name or path of the pre-trained model
    
    Returns:
        Tuple containing model, image processor, and device
    """
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    
    return model, image_processor


def keep_only_interesting_classes(predictions: List[Dict]) -> List[Dict]:
    """
    Keep only the classes of interest in the predictions.
    
    Args:
        predictions: Dictionary containing the predictions
    
    Returns:
        Dictionary containing only the classes of interest
    """
    global CONFIG

    for prediction in predictions:
        # Get the indices of the classes of interest
        interesting_ids = [id for id, label in CONFIG.id2label.items() if label in consts.INTERESTING_CLASSES]
        interesting_indices = [i for i, info in enumerate(prediction['segments_info']) if info['label_id'] in interesting_ids]

        # Filter out the classes of interest
        prediction['segments_info'] = [prediction['segments_info'][i] for i in interesting_indices]
        prediction['segmentation'] = np.where(np.isin(prediction['segmentation'], interesting_indices), prediction['segmentation'], -1)

    return predictions
    


def run_instance_segmentation(
        model: Mask2FormerForUniversalSegmentation, 
        image_processor: Mask2FormerImageProcessor, 
        images: Union[Image.Image, List[Image.Image]], 
        device: torch.device
    ) -> List[Dict]:
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
    
    inputs = image_processor(images=images, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model.forward(**inputs.to(device))

    # Crear el tensor de target_sizes
    img_size = torch.tensor([images[0].size[1], images[0].size[0]], dtype=torch.int32)  # Use int32 here
    target_sizes = img_size.repeat(len(images), 1).to(device)

    # Perform post-processing to get instance segmentation map
    predictions = image_processor.post_process_instance_segmentation(
        outputs, 
        target_sizes=[tuple(size.tolist()) for size in target_sizes]  # Ensure it's a tuple of ints
    )

    # Filter the predictions so we only have class 'car' and 'person'
    filtered_predictions = keep_only_interesting_classes(predictions)

    return filtered_predictions

def save_predictions(predictions, video_name: str, output_dir: str, batch_start_idx: int=0):
    """
    Convierte predicciones de Mask2Former a formato COCO RLE y las guarda en un archivo JSON.
    
    Args:
        predictions: Lista de predicciones para un batch de frames
        video_name: Nombre del video
        batch_start_idx: Índice del primer frame en el batch
        output_dir: Directorio para guardar los archivos
    """
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"preds_coco_{video_name}.json")
    
    # Cargar predicciones existentes o crear nueva lista
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            coco_predictions = json.load(f)
    else:
        coco_predictions = []
    
    # Procesar cada frame en el batch
    for batch_idx, frame_pred in enumerate(predictions):
        image_id = batch_start_idx + batch_idx
        
        # Extraer máscaras, clases y puntuaciones
        pred_masks = frame_pred['pred_masks'] 
        pred_classes = frame_pred['pred_classes']
        pred_scores = frame_pred['pred_scores']
        
        # Procesar cada instancia detectada
        for i in range(len(pred_scores)):
            # Convertir máscara a numpy si es tensor
            if isinstance(pred_masks[i], torch.Tensor):
                mask_np = pred_masks[i].cpu().numpy().astype(np.uint8)
            else:
                mask_np = pred_masks[i].astype(np.uint8)
            
            # Convertir a RLE
            rle = mask_util.encode(np.asfortranarray(mask_np))
            rle['counts'] = rle['counts'].decode('utf-8')  # Convertir bytes a string
            
            # Calcular bbox
            bbox = mask_util.toBbox(rle).tolist()  # [x, y, width, height]
            
            # Obtener category_id y score
            if isinstance(pred_classes[i], torch.Tensor):
                category_id = int(pred_classes[i].item())
            else:
                category_id = int(pred_classes[i])
                
            if isinstance(pred_scores[i], torch.Tensor):
                score = float(pred_scores[i].item())
            else:
                score = float(pred_scores[i])
            
            # Crear predicción COCO
            coco_pred = {
                'image_id': int(image_id),
                'category_id': category_id,
                'segmentation': rle,
                'score': score,
                'bbox': bbox
            }
            
            coco_predictions.append(coco_pred)
    
    # Guardar en formato JSON
    with open(output_file, 'w') as f:
        json.dump(coco_predictions, f)
    
    print(f"Predicciones guardadas para el vídeo {video_name} en {output_file}")


def visualize_prediction(
        image: Image.Image,
        predictions: Dict,
        output_path: str = 'segmentation_results/'
    ):
    """
    Visualize the segmentation predictions on the image and save as PNG.

    Args:
        image: The input image (PIL Image).
        predictions: Dictionary containing segmentation masks and class labels.
        output_path: Path to save the output image. If None, a default path is used.
    """
    # Convert the image to a numpy array for visualization
    image_np = np.array(image)

    # Extract masks and labels from predictions
    mask = predictions['segmentation']  # Shape: (num_objects, height, width)
    info = predictions['segments_info']  # Shape: (num_objects,)

    # Prepare the color map for visualization
    # Generate random colors for each instance
    num_instances = (mask.max() + 2).astype(int)
    colors = [tuple(np.random.randint(0, 255, 3)) for _ in range(num_instances)]

    # Create a blank mask to overlay
    overlay = np.zeros_like(image_np)

    # Loop through each instance to draw its mask
    for segment_info in info:
        i = segment_info['id']
        instance_mask = (mask == i)  # Binary mask for the current instance

        # Get the color for the current instance
        color = colors[i]

        # Apply the color to the instance mask
        overlay[instance_mask] = color

        # Optionally, add text (label and score) on top of the instance
        label = segment_info['label_id']
        score = segment_info['score']
        label_text = f'ID: {label}, Score: {score:.2f}'

        # Draw text on the image (this is optional)
        y, x = np.where(instance_mask)
        if len(y) > 0 and len(x) > 0:
            text_position = (x[0], y[0])  # Position the text at the first pixel of the mask
            cv2.putText(image_np, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Combine the original image with the overlay mask
    # You can adjust the alpha value for transparency if needed
    combined_image = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the output image
    output_file = os.path.join(output_path, 'segmentation_result.png')
    Image.fromarray(combined_image).save(output_file)

    print(f"Segmentation result saved to {output_file}")



if __name__ == '__main__':

    DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'
    #DATASET_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS'
    #DATASET_PATH = consts.KITTI_MOTS_PATH_ALEX

    # Get video names
    videos = os.listdir(os.path.join(DATASET_PATH, 'training/image_02'))
    videos = [video for video in videos if not video.startswith('.')]

    # Load model
    model, image_processor, device = load_model()
    data_loader = VideoDataset(DATASET_PATH)

    for video in tqdm(videos, desc="Processing videos", unit="video"):
        
        video_data = data_loader.load_video(video)
        frames = video_data['image']
        
        batch_size = 4
        batch_number = 1
        for i in tqdm(range(0, len(frames), batch_size)):
            
            batch_frames = frames[i:i + batch_size]
            predictions = run_instance_segmentation(model, image_processor, batch_frames, device=device)

            output_dir = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/week2/Evaluation_off-the-shelf/predictions'
            save_predictions(predictions, video, output_dir=output_dir, batch_start_idx=i)
            batch_number += 1
            
            # Visualize predictions
            for j, frame in enumerate(batch_frames):
                visualize_prediction(frame, predictions[j])

    print("Instance segmentation with Mask2Former off-the-shelf finished!")
