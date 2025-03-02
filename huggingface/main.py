import os
import torch
from typing import Dict
from pycocotools.mask import toBbox
import json
from coco_eval import CocoEvaluator
from inference import load_model, run_inference
from read_data import read_data
import consts
from pycocotools.coco import COCO


def load_images_and_annotations_for_video(video_folder: str, annotation_file: str) -> Dict:
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

        img_path = os.path.join(video_folder, f"{frame_id:06d}.png")
        if not os.path.exists(img_path):
            continue  # Skip missing images

        # Decode RLE mask
        rle = {'size': [h, w], 'counts': rle_mask.encode('utf-8')}
        bbox = toBbox(rle).tolist()

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
            'iscrowd': 0,
            'area': bbox[2] * bbox[3],
            'category_id': class_id,
            'bbox': bbox,
            'track_id': track_id,
        }
        coco_annotations.append(annotation)
        annotation_id_counter += 1
        
        # Add categories
        if class_id not in categories:
            if class_id == 1:
                category_name = 'car'
            # elif class_id == 2:
            #     category_name = 'pedestrian'
            else:
                category_name = 'unknown'
            
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
    # video0000_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/training/image_02/0000'
    # annotations_file = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/instances_txt/0000.txt'
    video0000_folder = f'{consts.KITTI_MOTS_PATH_ALEX}training/image_02/0000'
    annotations_file = f'{consts.KITTI_MOTS_PATH_ALEX}instances_txt/0000.txt'
    output_json_path = './gt_coco_0000.json'

    gt_coco = load_images_and_annotations_for_video(video0000_folder, annotations_file)
    
    # Save the gt_coco as a JSON file
    #output_json_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/gt_coco_0000.json'
    with open(output_json_path, 'w') as f:
        json.dump(gt_coco, f, indent=4)
    print(f"Ground truth COCO annotations saved to: {output_json_path}")

    gt_coco = COCO(output_json_path)

    coco_evaluator = CocoEvaluator(gt_coco)

    # Load dataset
    dataset = read_data(consts.KITTI_MOTS_PATH_ALEX)
    dataset = dataset['train']['image']

    model, image_processor, device = load_model()
    results = []
    for i in range(0, len(dataset), 10):
        batch = dataset[i:min(i + 10, len(dataset))]
        results += run_inference(model, image_processor, batch, device)

    formatted_results = {}
    for i, (image, result) in enumerate(zip(dataset, results)):
        formatted_results[i + 1] = {
            'image': image,
            'labels': result["labels"],
            'boxes': result["boxes"],
            'scores': result["scores"],
        }

    coco_evaluator.update(formatted_results)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    