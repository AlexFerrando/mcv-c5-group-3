import json
import cv2
import os

def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x, y, w, h = map(int, box['bbox'])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {box['category_id']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def create_video(gt_file, pred_file, output_video):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    image_info = {img['id']: img['file_name'] for img in gt_data['images']}
    gt_annotations = {img_id: [] for img_id in image_info.keys()}
    for ann in gt_data['annotations']:
        gt_annotations[ann['image_id']].append(ann)

    pred_annotations = {img_id: [] for img_id in image_info.keys()}
    for ann in pred_data:
        pred_annotations[ann['image_id']].append(ann)

    frame_size = (1242, 375)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 10, frame_size)

    for img_id, file_name in image_info.items():
        image = cv2.imread(file_name)
        draw_boxes(image, gt_annotations[img_id], (0, 255, 0), 'GT')
        draw_boxes(image, pred_annotations[img_id], (0, 0, 255), 'Pred')
        out.write(image)

    out.release()

# Uso de la función
# gt_file = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/gt_coco_0000.json'
# pred_file = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/predictions_10.json'
# output_video = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/output_video.mp4'

gt_file = '/home/alex/Documents/MCV/C5/mcv-c5-group-3/gt_coco_0000.json'
pred_file = '/home/alex/Documents/MCV/C5/mcv-c5-group-3/coco_results.json'
output_video = '/home/alex/Documents/MCV/C5/mcv-c5-group-3/output_video.mp4'

create_video(gt_file, pred_file, output_video)