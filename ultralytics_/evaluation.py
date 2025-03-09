import os
import argparse

from ultralytics import YOLO
import wandb

import consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model")
    parser.add_argument("--output_path", type=str, help="Path to save the results", default="/ghome/c5mcv03/mcv-c5-group-3/outputs/pol/job_outputs")
    parser.add_argument("--split_val", type=bool, help="Evaluate the validation set", default=True)
    args = parser.parse_args()

    wandb.login(key = "8410a2da3f323633210ca8d25ce6862368d9f489")
    model = YOLO(args.model_path)
    results = model.val(data=os.path.join(consts.PATH_KITTI_MOTS_YOLO, "kitti_mots_config.yaml"),
                        classes=consts.YOLO_CLASSES, cache=False, project=args.output_path,
                        split="val" if args.split_val else "train",
                        name = "off the shelf hybrid outputs pretrained model yolo11n")
    print(results.box.map)  # mAP50-95
    print(results.box.map50)  # mAP50
    print(results.box.map75)  # mAP75
    print(results.box.maps)  # list of mAP50-95 for each category
    # Save the evaluation metrics to a file
    metrics_file = os.path.join(args.output_path, "evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"mAP50-95: {results.box.map}\n")
        f.write(f"mAP50: {results.box.map50}\n")
        f.write(f"mAP75: {results.box.map75}\n")
        f.write("mAP50-95 per category:\n")
    print(f"Metrics saved to {metrics_file}")