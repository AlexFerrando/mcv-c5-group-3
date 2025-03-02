import os
from typing import Dict, List
from pycocotools.mask import toBbox
import json


def load_images_and_annotations_for_video(video_folder: str, annotation_file: str, target_classes: List[int] = [1, 2]) -> Dict:
    """
    Load annotations and convert to COCO format, filtering for specific object classes.
    
    Args:
        video_folder: Path to the folder containing image frames
        annotation_file: Path to the annotation file
        target_classes: List of class IDs to include (default: [1, 2] for 'car' and 'pedestrian')
    
    Returns:
        Dictionary in COCO format with filtered annotations
    """
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
            
        img_path = os.path.join(video_folder, f"{frame_id:06d}.png")
        if not os.path.exists(img_path):
            continue  # Skip missing images

        # Decode RLE mask
        rle = {'size': [h, w], 'counts': rle_mask.encode('utf-8')}
        bbox = toBbox(rle).tolist()
        
        # Calculate area from bbox [x, y, width, height]
        area_value = bbox[2] * bbox[3]  # width * height

        # Create or update the image entry
        if frame_id not in frame_to_mask:
            frame_to_mask[frame_id] = {
                "image": img_path,
                "frame_id": frame_id,
                "track_id": [],
                "class_id": [],
                "bbox": []
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
            'file_name': frame_data['image'],
            'width': w,  # Width from the RLE
            'height': h,  # Height from the RLE
        })

    return coco_format


if __name__ == '__main__':
    video0000_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/training/image_02/0000'
    annotations_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/instances_txt/0000.txt'

    gt_coco = load_images_and_annotations_for_video(video0000_folder, annotations_folder)
    
    # Save the gt_coco as a JSON file
    output_json_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/gt_coco_0000.json'
    with open(output_json_path, 'w') as f:
        json.dump(gt_coco, f, indent=4)
    print(f"Ground truth COCO annotations saved to: {output_json_path}")
