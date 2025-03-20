import os
from typing import Dict, List
from pycocotools.mask import toBbox
import json

def load_images_and_annotations_for_video(video_name: str, target_classes: List[int] = [1, 2]) -> Dict:
    """
    Load annotations and convert to COCO format, filtering for specific object classes.
    
    Args:
        video_name: Path to the folder containing image frames
        target_classes: List of class IDs to include (default: [1, 2] for 'car' and 'pedestrian')
    
    Returns:
        Dictionary in COCO format with filtered annotations
    """

    video_folder = DATASET_PATH+f'/training/image_02/{video_name}'
    annotation_file = DATASET_PATH+f'/instances_txt/{video_name}.txt'

    with open(annotation_file, "r") as f:
        annotations = f.readlines()

    frame_to_mask = {}
    coco_annotations = []
    image_id_counter = 1
    annotation_id_counter = 1
    categories = {}
    
    for line in annotations:
        parts = line.strip().split(" ")
        frame_id, track_id, class_id, h, w = map(int, parts[:5])
        rle_mask = " ".join(parts[5:])

        # Skip annotations for classes we're not interested in
        if class_id not in target_classes:
            continue
            
        # Create RLE dictionary in formato correcto para COCO
        rle = {'size': [h, w], 'counts': rle_mask}  # Ya está como string, no necesita encode/decode
        bbox = toBbox(rle).tolist()
        
        # Calcular área desde bbox [x, y, width, height]
        area_value = bbox[2] * bbox[3]  # width * height

        # Create or update the image entry
        if frame_id not in frame_to_mask:
            frame_to_mask[frame_id] = {
                #"image": img_path,
                "frame_id": frame_id,
                "track_id": [],
                "class_id": [],
                "bbox": [],
                "masks": []
            }
            # Add image metadata for COCO format
            frame_to_mask[frame_id]["image_id"] = image_id_counter
            image_id_counter += 1
            
        # Store frame annotation
        frame_to_mask[frame_id]["track_id"].append(track_id)
        frame_to_mask[frame_id]["class_id"].append(class_id)
        frame_to_mask[frame_id]["bbox"].append(bbox)

        # Add annotations in COCO format
        annotation = {
            'id': annotation_id_counter,
            'image_id': frame_to_mask[frame_id]["image_id"],
            'category_id': class_id,
            'bbox': bbox,
            'track_id': track_id,
            'area': area_value,       # Required field: area of the bounding box
            'iscrowd': 0,             # Required field: 0 indicates it's not a crowd
            'segmentation': rle       # Usar directamente el RLE (no decodificado)
        }
        coco_annotations.append(annotation)
        annotation_id_counter += 1
        
        # Add categories
        if class_id not in categories:
            if class_id == 1:
                category_name = 'car'
            elif class_id == 2:
                category_name = 'pedestrian'
            
            categories[class_id] = {
                'id': class_id,
                'name': category_name
            }

    # COCO format output
    coco_format = {
        'images': [],
        'annotations': coco_annotations,
        'categories': list(categories.values())
    }

    # Add image metadata for all frames
    for frame_id, frame_data in frame_to_mask.items():
        coco_format['images'].append({
            'id': frame_data['image_id'],
            #'file_name': frame_data['image'],
            'width': w,  # Width from the RLE
            'height': h,  # Height from the RLE
        })

    return coco_format


def save_GroundTruth(predictions: List[Dict], video_name: str) -> None:
    
    # Definir la ruta de salida
    output_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/week2/evaluation_off-the-shelf/ground_truth'
    os.makedirs(output_folder, exist_ok=True)  # Crea la carpeta si no existe

    # Definir la ruta del archivo
    output_json_path = f'{output_folder}/gt_coco_{video_name}.json'

    # Guardar el JSON
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)


if __name__ == '__main__':

    DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'

    # Get video names
    files = os.listdir(DATASET_PATH+'/instances_txt')
    video_names = [filename.split('.')[0] for filename in files]

    # Reand and save the GT in COCO format
    for video_name in video_names:
        gt_coco = load_images_and_annotations_for_video(video_name)
        
        save_GroundTruth(gt_coco, video_name)
    
    print(f"Ground truth saved in COCO format for all the videos!")
