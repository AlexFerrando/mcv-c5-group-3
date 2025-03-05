from detectron2.data import DatasetCatalog

import os

import consts
from detector import *
from read_data import register_kitti_mots

def main():

    # Load data
    dataset_path = consts.KITTI_MOTS_PATH
    register_kitti_mots(dataset_path)

    dataset_name = "kitti_mots_training"
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Load model
    detector = Detector()

    # Output directory
    output_dir = "detectron2_inference"
    os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(dataset_dicts[:10]):

        image_path = sample["file_name"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = detector.run_inference(Image.fromarray(image))

        # Generate output filename
        output_path = os.path.join(output_dir, f"detection_{idx}.png")

        # Visualize and save detections
        detector.visualize_detections(Image.fromarray(image), results, output_path)

if __name__ == '__main__':
    main()