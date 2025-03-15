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

def load_model(model_name: str = consts.MASK2FORMER, modified: bool = False) -> Tuple[Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor]:
    """
    Load model, processor, and determine device.
    
    Args:
        model_name: Name or path of the pre-trained model
    
    Returns:
        Tuple containing model, image processor, and device
    """
    image_processor = Mask2FormerImageProcessor.from_pretrained(
        model_name,
        # size = {"width": 480, "height": 480}, # se puede cambiar a otro tamaño
    )
    if modified:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            label2id=consts.LABEL2ID,
            id2label=consts.ID2LABEL,
            ignore_mismatched_sizes=True
        )
    else:
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

    
def filter_predictions_by_mask(predictions):
    """
    Filtra las predicciones para incluir solo objetos que tienen representación
    en el mapa de segmentación.
    """
    filtered_results = []
    
    for pred in predictions:
        # Obtener IDs presentes en la máscara de segmentación
        valid_ids = set(np.unique(pred['segmentation']).tolist())
        
        # Eliminar -1 que representa el fondo
        if -1 in valid_ids:
            valid_ids.remove(-1)

        
        # Filtrar segments_info
        filtered_segments = [info for info in pred['segments_info'] 
                             if info['id'] in valid_ids]
        
        # Crear predicción filtrada
        filtered_pred = {
            'segmentation': pred['segmentation'],
            'segments_info': filtered_segments
        }
        
        filtered_results.append(filtered_pred)
    
    return filtered_results


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
        target_sizes=[tuple(size.tolist()) for size in target_sizes],
        mask_threshold=0.3  # Ensure it's a tuple of ints
    )

    # Filter the predictions so we only have class 'car' and 'person'
    filtered_predictions_byCLASS = keep_only_interesting_classes(predictions)
    
    # There are some detected objects that don't appear in the mask. Filter them out.
    filtered_predictions_byMASK = filter_predictions_by_mask(filtered_predictions_byCLASS)

    return filtered_predictions_byMASK


def coco_reformat(predictions_list):
    """
    Convierte predicciones de Mask2Former a formato COCO para evaluación.
    
    Args:
        predictions_list: Lista de diccionarios con predicciones para cada frame
                         (cada uno contiene 'segmentation' y 'segments_info')
    
    Returns:
        List[Dict]: Lista de diccionarios en formato COCO
    """
    
    coco_results = []
    
    # Procesar cada frame
    for frame_idx, frame_pred in enumerate(predictions_list, 1):  # Empezando desde 1 como en la segunda función
        # Obtener mapa de segmentación y segments_info
        segmentation_map = frame_pred['segmentation']
        segments_info = frame_pred['segments_info']
        
        # Identificar IDs presentes en el mapa de segmentación
        if isinstance(segmentation_map, torch.Tensor):
            segmentation_map = segmentation_map.cpu().numpy()
        unique_ids = np.unique(segmentation_map)
        # Eliminar ID de fondo (-1) si existe
        unique_ids = unique_ids[unique_ids != -1] if -1 in unique_ids else unique_ids
 
        
        # Para cada instancia en segments_info
        for segment in segments_info:
            segment_id = segment['id']
            
            # Filtrar objetos que no tienen representación en el mapa de segmentación
            if segment_id not in unique_ids:
                continue
            
            # Crear máscara binaria para este segmento
            binary_mask = (segmentation_map == segment_id).astype(np.uint8)
            
            # Convertir a formato RLE
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            # Convertir counts de bytes a string para JSON
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            
            # Obtener category_id y score
            category_id = segment['label_id']
            score = segment['score']
            
            # Calcular bbox si no está disponible
            if 'bbox' in segment:
                bbox = segment['bbox']
            else:
                bbox = mask_util.toBbox(rle).tolist()  # [x, y, width, height]
            
            # Crear entrada en formato COCO
            coco_result = {
                'image_id': int(frame_idx),
                'category_id': consts.inverse_mapping_class_id('coco', int(category_id)),
                'segmentation': rle,  # Incluimos el RLE para evaluación de máscaras
                'bbox': bbox,
                'score': float(score)
            }
            
            coco_results.append(coco_result)
    
    return coco_results


def save_predictions(coco_results, video_name, output_dir="predictions"):
    """
    Guarda predicciones en formato COCO a un archivo JSON.
    
    Args:
        coco_results: Lista de diccionarios en formato COCO
        video_name: Nombre del video para nombrar el archivo
        output_dir: Directorio donde guardar el archivo JSON
    
    Returns:
        str: Ruta del archivo guardado
    """    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"preds_coco_{video_name}.json")
    
    # Guardar resultados en JSON
    with open(output_file, 'w') as f:
        json.dump(coco_results, f, indent=4)

    print(f"Guardadas {len(coco_results)} predicciones para el vídeo {video_name} en {output_file}")
    
    return output_file


def visualize_prediction(
        image: Image.Image,
        predictions: Dict,
        file_name: str,
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
    output_path = 'segmentation_results/segmentation_result'
    os.makedirs(output_path, exist_ok=True)

    # Save the output image
    output_file = os.path.join(output_path, file_name)
    Image.fromarray(combined_image).save(output_file)

    print(f"Segmentation result saved to {output_file}")



if __name__ == '__main__':

    #DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'
    DATASET_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS'
    #DATASET_PATH = consts.KITTI_MOTS_PATH_ALEX

    # Get video names
    videos = os.listdir(os.path.join(DATASET_PATH, 'training/image_02'))
    videos = [video for video in videos if not video.startswith('.')]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor = load_model()
    model.to(device)
    data_loader = VideoDataset(DATASET_PATH)

    for video in tqdm(videos, desc="Processing videos", unit="video"):
        
        video_data = data_loader.load_video(video)
        frames = video_data['image']
        
        batch_size = 10
        predictions_video = []

        for i in tqdm(range(0, len(frames), batch_size)):
            batch_frames = frames[i:i + batch_size]
            predictions_batch = run_instance_segmentation(model, image_processor, batch_frames, device=device)
            predictions_video.extend(predictions_batch)
            
            # # Visualize predictions
            # for j, frame in enumerate(batch_frames):
            #     visualize_prediction(frame, predictions_batch[j], file_name=f"{video}_{i+j}.png")
        
        # After the hole video is processed, save the predictions in COCO format into a JSON file
        predictions_coco = coco_reformat(predictions_video)
        #output_dir = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/week2/Evaluation_off-the-shelf/predictions'
        output_dir = '/ghome/c5mcv03/mcv-c5-group-3/huggingface/week2/preds_off-the-shelf'
        save_predictions(predictions_coco, video, output_dir=output_dir)

    print("Instance segmentation with Mask2Former off-the-shelf finished!")
