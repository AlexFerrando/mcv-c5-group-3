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
    parser.add_argument("--val_name", type=str, help="Name for the validation results", default="off the shelf hybrid outputs pretrained model yolo11n")
    args = parser.parse_args()

    wandb.login(key = "8410a2da3f323633210ca8d25ce6862368d9f489")
    model = YOLO(args.model_path)
    results = model.val(data=os.path.join(consts.PATH_KITTI_MOTS_YOLO_ALL, "kitti_mots_config.yaml"),
                        classes=consts.YOLO_CLASSES, cache=False, project=args.output_path,
                        # split="val" if args.split_val else "train",
                        name=args.val_name)
    print(results.box.map)  # mAP50-95
    print(results.box.map50)  # mAP50
    print(results.box.map75)  # mAP75
    print(results.box.maps)  # list of mAP50-95 for each category
    # Save the evaluation metrics to a file
    metrics_file = os.path.join(args.output_path, f"{args.val_name}_evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"mAP50-95: {results.box.map}\n")
        f.write(f"mAP50: {results.box.map50}\n")
        f.write(f"mAP75: {results.box.map75}\n")
        f.write("mAP50-95 per category:\n")
        f.write(f"    person: {results.box.maps[0]}\n")
        f.write(f"    car: {results.box.maps[2]}\n")
        f.write("\n\n")
        f.write("    All categories:\n")
        f.write(f"    {results.box.maps}\n")
        #TODO include
    # Print specific metrics
    print("Class indices with average precision:", results.ap_class_index)
    print("Average precision for all classes:", results.box.all_ap)
    print("Average precision:", results.box.ap)
    print("Average precision at IoU=0.50:", results.box.ap50)
    print("Class-specific results:", results.box.class_result)
    print("F1 score:", results.box.f1)
    print("Mean average precision:", results.box.map)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean average precision at IoU=0.75:", results.box.map75)
    print("Mean precision:", results.box.mp)
    print("Mean recall:", results.box.mr)
    print("Precision:", results.box.p)
    print("Recall:", results.box.r)
        
        
    print(f"Metrics saved to {metrics_file}")