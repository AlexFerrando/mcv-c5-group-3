import os
import numpy as np
from pycocotools import mask as COCOmask
from pycocotools.mask import toBbox
import matplotlib.pyplot as plt
import shutil

KITTI2COCO = {
    1: 2, # Car
    2: 0, # Pedestrian
}

def decode_rle_batch(rle_list):
    decoded_masks = COCOmask.decode(rle_list)
    return decoded_masks

def extract_yolo_annotations(ann_filepath):
    print(f"Processing: {ann_filepath}")
    with open(ann_filepath) as file:
        objs = file.readlines()
    
    obj_timeframes = []
    img_height, img_width = 0, 0  # Placeholder values to be updated dynamically

    rle_list = []
    obj_info_list = []

    for obj in objs:
        obj_info = obj.split()
        obj_class_id = int(obj_info[2])
        if obj_class_id == 10:
            continue
        obj_timeframes.append(obj_info[0])
        print(obj_info[0], end=' ')
        img_height = int(obj_info[3])  # Assuming the height is stored at index 3
        img_width = int(obj_info[4])   # Assuming the width is stored at index 4
        rle_str = obj_info[5]
        
        # Prepare RLE dictionary for batch decoding
        rle = {
            'size': [img_height, img_width],
            'counts': rle_str
        }
        rle_list.append(rle)
        obj_info_list.append((KITTI2COCO[obj_class_id], img_width, img_height))
    print()
    # Decode RLE masks in batch
    try:
        decoded_masks = decode_rle_batch(rle_list)
    except Exception as e:
        print(f"Error decoding RLE batch: {e}")
        return []

    yolo_annotations = {}

    print("shape decoded_masks", decoded_masks.shape)
    for i in range(decoded_masks.shape[2]):
        obj_timeframe = obj_timeframes[i].zfill(6)
        if obj_timeframe not in yolo_annotations:
            yolo_annotations[obj_timeframe] = []
        obj_class_id, img_width, img_height = obj_info_list[i]
        decoded_mask = decoded_masks[:, :, i]
        # Calculate bounding box from the decoded mask
        y_indices, x_indices = decoded_mask.nonzero()
        if len(x_indices) == 0 or len(y_indices) == 0:
            print(f"Empty mask for object {i}, skipping.")
            continue
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        x_center = (x_min + bbox_width / 2) / img_width
        y_center = (y_min + bbox_height / 2) / img_height
        bbox_width /= img_width
        bbox_height /= img_height
        yolo_annotation = f"{obj_class_id} {x_center} {y_center} {bbox_width} {bbox_height}"
        print(obj_timeframe, end=' ')
        yolo_annotations[obj_timeframe].append(yolo_annotation)
    print()
    return yolo_annotations

def convert_kitti_to_yolo(kitti_anno_dir, output_dir, copy_images=False):
    os.makedirs(output_dir, exist_ok=True)
    output_dir_images = os.path.join(output_dir, "images")
    os.makedirs(output_dir_images, exist_ok=True)
    output_dir_labels = os.path.join(output_dir, "labels")
    os.makedirs(output_dir_labels, exist_ok=True)
    for ann_filename in sorted(os.listdir(kitti_anno_dir)):
        ann_filepath = os.path.join(kitti_anno_dir, ann_filename)
        yolo_annotations = extract_yolo_annotations(ann_filepath)
        
        # Save YOLO annotations to file
        if yolo_annotations:
            for yolo_img_label, yolo_img_objects in yolo_annotations.items():
                sequence = os.path.basename(ann_filename).replace('.txt', '')
                yolo_ann_filename = f"{sequence}_{yolo_img_label}.txt"
                yolo_ann_filepath = os.path.join(output_dir_labels, yolo_ann_filename)
                with open(yolo_ann_filepath, 'w') as f:
                    f.write('\n'.join(yolo_img_objects))
                print(f"Saved YOLO annotations to {yolo_ann_filepath}")
        if copy_images:
            # Copy images to the output directory
            img_dir = os.path.join(kitti_anno_dir, "..", "training", "image_02")
            for root, _, files in os.walk(img_dir):
                for file in files:
                    if file.endswith(".png") or file.endswith(".jpg"):
                        folder_name = os.path.basename(root)
                        new_file_name = f"{folder_name}_{file}"
                        src_file_path = os.path.join(root, file)
                        dst_file_path = os.path.join(output_dir_images, new_file_name)
                        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                        shutil.copy(src_file_path, dst_file_path)
                        print(f"Copied {src_file_path} to {dst_file_path}")
        
if __name__ == "__main__":
    # kitti_dir_input = "/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS/instances_txt"
    kitti_dir_input = "/projects/master/c5/KITTI_MOTS/instances_txt"
    # kitti_dir_output = "/ghome/c5mcv03/test/KITTI-MOTS-YOLO"
    kitti_dir_output = "/projects/master/c5/test"
    convert_kitti_to_yolo(kitti_dir_input, kitti_dir_output, copy_images=True)