import os
from pycocotools.mask import toBbox
import shutil

KITTI2COCO = {
    1: 2, # Car
    2: 0, # Pedestrian
}

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
        assert obj_class_id in KITTI2COCO, f"Unknown class ID: {obj_class_id}"
        obj_timeframes.append(obj_info[0])
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
    try:
        # decoded_masks = decode_rle_batch(rle_list)
        decoded_bboxes = toBbox(rle_list)
    except Exception as e:
        print(f"Error decoding RLE batch: {e}")
        return []

    yolo_annotations = {}

    # Iterate over each bounding box to compute YOLO format values
    for i, bbox in enumerate(decoded_bboxes):
        assert len(decoded_bboxes) == len(obj_timeframes) == len(obj_info_list)
        obj_timeframe = obj_timeframes[i].zfill(6)
        if obj_timeframe not in yolo_annotations:
            yolo_annotations[obj_timeframe] = []
        obj_class_id, img_width, img_height = obj_info_list[i]
        
        # Use the bounding box values directly
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        yolo_annotation = f"{obj_class_id} {x_center} {y_center} {w_norm} {h_norm}"
        yolo_annotations[obj_timeframe].append(yolo_annotation)
    return yolo_annotations

def convert_kitti_to_yolo(kitti_anno_dir, output_dir, training_sequences, validation_sequences, copy_images=False):
    def process_kitti_annotations(kitti_anno_dir, output_dir, sequence_ids, copy_images, folder_name):
        output_dir = os.path.join(output_dir, folder_name)
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
                            # print(f"Copied {src_file_path} to {dst_file_path}")

    # Call the function for training and validation sequences
    process_kitti_annotations(kitti_anno_dir, output_dir, training_sequences, copy_images, "training")
    process_kitti_annotations(kitti_anno_dir, output_dir, validation_sequences, copy_images, "validation")
    # process_kitti_annotations(kitti_anno_dir, output_dir, training_sequences + validation_sequences, copy_images, "all")


if __name__ == "__main__":
    kitti_dir_input = "/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS/instances_txt"
    # kitti_dir_input = "/projects/master/c5/KITTI_MOTS/instances_txt"
    kitti_dir_output = "/ghome/c5mcv03/test/KITTI-MOTS-YOLO"
    # kitti_dir_output = "/projects/master/c5/test/KITTI_MOTS_YOLO"

    training_sequences = [str(sequence).zfill(4) for sequence in list(range(0, 16))]
    validation_sequences = [str(sequence).zfill(4) for sequence in list(range(16, 21))]
    
    convert_kitti_to_yolo(kitti_anno_dir = kitti_dir_input,
                          output_dir = kitti_dir_output,
                          training_sequences = training_sequences,
                          validation_sequences = validation_sequences,
                          copy_images = True)