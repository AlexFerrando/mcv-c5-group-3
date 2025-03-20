import os
import argparse

from ultralytics import YOLO
import wandb

import consts


def validate_model(model, output_path, split_val=True, val_name="off-the-shelf evaluation"):
    results = model.val(data=os.path.join(consts.PATH_KITTI_MOTS_YOLO_SEGMENTATION_ALL, "kitti_mots_config.yaml"),
                        classes=consts.YOLO_CLASSES, cache=False, project=output_path,
                        split=split_val,
                        name=val_name)

    metrics_file = os.path.join(output_path, f"{val_name}_evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"mAP50-95: {results.seg.map}\n")
        f.write(f"mAP50: {results.seg.map50}\n")
        f.write(f"mAP75: {results.seg.map75}\n")
        f.write("mAP50-95 per category:\n")
        f.write(f"    person: {results.seg.maps[0]}\n")
        f.write(f"    car: {results.seg.maps[2]}\n")
        f.write("\n\n")
        f.write("    All categories:\n")
        f.write(f"    {results.seg.maps}\n")
        f.write("Class indices with average precision:\n")
        f.write(f"    {results.ap_class_index}\n")
        f.write("Average precision for all classes:\n")
        f.write(f"    {results.seg.all_ap}\n")
        f.write(f"Average precision: {results.seg.ap}\n")
        f.write(f"Average precision at IoU=0.50: {results.seg.ap50}\n")
        f.write("Class-specific results:\n")
        f.write(f"    {results.seg.class_result}\n")
        f.write(f"F1 score: {results.seg.f1}\n")
        f.write(f"Mean average precision: {results.seg.map}\n")
        f.write(f"Mean average precision at IoU=0.50: {results.seg.map50}\n")
        f.write(f"Mean average precision at IoU=0.75: {results.seg.map75}\n")
        f.write(f"Mean precision: {results.seg.mp}\n")
        f.write(f"Mean recall: {results.seg.mr}\n")
        f.write(f"Precision: {results.seg.p}\n")
        f.write(f"Recall: {results.seg.r}\n")
    print(f"Metrics saved to {metrics_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model")
    parser.add_argument("--output_path", type=str, help="Path to save the results", default="/ghome/c5mcv03/mcv-c5-group-3/outputs/pol/job_outputs")
    parser.add_argument("--split_val", type=str, help="Evaluate the validation set", default="val")
    parser.add_argument("--val_name", type=str, help="Name for the validation results", default="off-the-shelf evaluation")
    args = parser.parse_args()

    wandb.login(key = "8410a2da3f323633210ca8d25ce6862368d9f489")
    model = YOLO(args.model_path)
    validate_model(model, args.output_path, args.split_val, args.val_name)
