import os
import argparse

from ultralytics import YOLO

from ultralytics_ import consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model")
    parser.add_argument("--output_path", type=str, help="Path to save the results", default="/projects/master/c5/outputs/results")
    parser.add_argument("--split_val", type=bool, help="Split the validation set", default=True)
    args = parser.parse_args()

    model = YOLO(args.model_path)
    results = model.val(data=os.path.join(consts.PATH_KITTI_MOTS_YOLO, "kitti_mots_config.yaml"),
                        classes=consts.YOLO_CLASSES, cache=False, project=args.output_path)
