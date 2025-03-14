import os

from pycocotools import mask as maskUtils
import numpy as np
import shutil
import cv2

import consts

def extract_yolo_annotations(ann_filepath):
    print(f"Processing: {ann_filepath}")

    with open(ann_filepath) as file:
        objs = file.readlines()

    yolo_annotations = {}
    for obj in objs:
        obj_info = obj.split()
        
        assert len(obj_info) == 6, f"Invalid object info: {obj_info}"
        assert class_id in consts.KITTI2COCO, f"Unknown class ID: {class_id}"

        timeframe_id, _, class_id, img_height, img_width = map(int, obj_info[:5])

        if class_id == 10:
            continue

        rle_str = obj_info[5]
        rle = {
            'size': [img_height, img_width],
            'counts': rle_str
        }

        decoded_mask = maskUtils.decode(rle)
        mask, _ = cv2.findContours(decoded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(mask) == 1, f"Invalid mask: {mask}"

        formatted_timeframe = f"{timeframe_id:06d}"
        if formatted_timeframe not in yolo_annotations:
            yolo_annotations[formatted_timeframe] = []

        yolo_annotations[formatted_timeframe].append(polygon_to_yolo_str(mask[0] / [img_width, img_height], consts.KITTI2COCO[class_id]))
    return yolo_annotations

def polygon_to_yolo_str(polygon_points: np.ndarray, class_index: int, precision: int = 4) -> str:
    # Flatten the array so that it becomes a 1D list of coordinates.
    flat_coords = polygon_points.flatten()
    coords_str = " ".join(f"{coord:.{precision}f}" for coord in flat_coords)
    return f"{class_index} {coords_str}"


def convert_kitti_to_yolo(kitti_anno_dir, output_dir, training_sequences, validation_sequences, copy_images=False):
    def process_kitti_annotations(kitti_anno_dir, output_dir, sequence_ids, copy_images, folder_name):
        output_dir = os.path.join(output_dir, folder_name)
        # Remove the output directory if it already exists for a fresh start
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        output_dir_images = os.path.join(output_dir, "images")
        os.makedirs(output_dir_images, exist_ok=True)
        output_dir_labels = os.path.join(output_dir, "labels")
        os.makedirs(output_dir_labels, exist_ok=True)
        
        for ann_filename in sorted(os.listdir(kitti_anno_dir)):
            if ann_filename.replace('.txt', '') not in sequence_ids:
                continue
            ann_filepath = os.path.join(kitti_anno_dir, ann_filename)
            yolo_annotations = extract_yolo_annotations(ann_filepath)

            # Save YOLO annotations to file
            if yolo_annotations:
                for yolo_img_timeframe_label, yolo_img_objects in yolo_annotations.items():
                    sequence = os.path.basename(ann_filename).replace('.txt', '')
                    os.makedirs(os.path.join(output_dir_labels, sequence), exist_ok=True)
                    yolo_ann_filename = f"{yolo_img_timeframe_label}.txt"
                    yolo_ann_filepath = os.path.join(output_dir_labels, yolo_ann_filename)
                    # TODO: check
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
                            if folder_name not in sequence_ids:
                                continue
                            ouput_folder_image_path = os.path.join(output_dir_images, folder_name)
                            os.makedirs(ouput_folder_image_path, exist_ok=True)
                            src_file_path = os.path.join(root, file)
                            dst_file_path = os.path.join(output_dir_images, file)
                            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                            shutil.copy(src_file_path, dst_file_path)

    # Call the function for training and validation sequences
    process_kitti_annotations(kitti_anno_dir, output_dir, training_sequences, copy_images, "training")
    process_kitti_annotations(kitti_anno_dir, output_dir, validation_sequences, copy_images, "validation")
    process_kitti_annotations(kitti_anno_dir, output_dir, validation_sequences, copy_images, "test")
    


if __name__ == "__main__":
    kitti_dir_input = "/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS/instances_txt"
    kitti_dir_input = "/projects/master/c5/KITTI-MOTS-big/instances_txt"
    # kitti_dir_output = "/projects/master/c5/test/KITTI_MOTS_YOLO"

    training_sequences = [str(sequence).zfill(4) for sequence in list(range(0, 16))]
    validation_sequences = [str(sequence).zfill(4) for sequence in list(range(16, 21))]
    

    convert_kitti_to_yolo(kitti_anno_dir = kitti_dir_input,
                          output_dir = consts.PATH_KITTI_MOTS_YOLO_SEGMENTATION,
                          training_sequences = training_sequences,
                          validation_sequences = validation_sequences,
                          copy_images = True)