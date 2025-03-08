from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from tqdm import tqdm
import sys
import re
import numpy as np


def per_video_evaluation():
    
    DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'
    
    # Get the list of video names
    videos = os.listdir(DATASET_PATH+'/training/image_02')[1:]

    # Output file for per-video results
    per_video_results_path = "/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/evaluation/per_video_metrics.txt"

    with open(per_video_results_path, "w") as per_video_file:
        per_video_file.write("Per-video COCO evaluation results:\n\n")

    for video_name in tqdm(videos, desc="Processing videos", unit="video"):

        video_gt = f'/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/ground_truth/gt_coco_{video_name}.json'
        video_preds = f'/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/predictions/preds_coco_{video_name}.json'

        try:
            # Load ground truth annotations and predictions for the current video
            coco_gt = COCO(video_gt)
            coco_dt = coco_gt.loadRes(video_preds)

            # Create a new COCOeval object for each video
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

            # Evaluate the current video
            coco_eval.evaluate()
            coco_eval.accumulate()

            # Append results to the per-video file
            with open(per_video_results_path, "a") as per_video_file:
                original_stdout = sys.stdout  # Save original stdout
                sys.stdout = per_video_file  # Redirect stdout to file
                
                per_video_file.write(f"\n=== Video: {video_name} ===\n")
                coco_eval.summarize()

                sys.stdout = original_stdout  # Restore original stdout

        except Exception as e:
            print(f"Error processing {video_name}: {e}")

    print(f"Per-video evaluation metrics saved to {per_video_results_path}")


def compute_mean_metrics_from_txt(file_path):
    """
    Reads a COCO evaluation results file and computes the mean for each metric across all videos.
    
    :param file_path: Path to the .txt file containing per-video COCO metrics.
    :return: A dictionary with the mean values of each metric.
    """
    # Regular expression pattern to extract metric values
    metric_pattern = re.compile(r'(.+?) = ([\d\.]+)')

    # Dictionary to store metric values
    metrics_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            match = metric_pattern.search(line)
            if match:
                metric_name = match.group(1).strip()
                metric_value = float(match.group(2))

                # Store metric values in a list for averaging later
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(metric_value)

    # Compute mean for each metric
    mean_metrics = {metric: np.mean(values) for metric, values in metrics_dict.items()}

    # Save results to a new file
    output_file = "global_metrics.txt"
    with open(output_file, "w") as out_file:
        out_file.write("Global COCO evaluation metrics (mean over all videos):\n\n")
        for metric, mean_value in mean_metrics.items():
            out_file.write(f"{metric}: {mean_value:.4f}\n")

    print(f"\nGlobal mean metrics saved to {output_file}")

    return mean_metrics


def per_class_evaluation():
    """
    Evaluates COCO metrics per class for each video and saves the results in a text file.
    """

    DATASET_PATH = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS'
    
    # Get the list of video names
    videos = os.listdir(DATASET_PATH+'/training/image_02')[1:]

    # Output file for per-class results
    per_class_results_path = "/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/evaluation/per_class_metrics.txt"

    with open(per_class_results_path, "w") as per_class_file:
        per_class_file.write("Per-class COCO evaluation results:\n\n")

    for video_name in tqdm(videos, desc="Processing videos", unit="video"):

        video_gt = f'/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/ground_truth/gt_coco_{video_name}.json'
        video_preds = f'/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/predictions/preds_coco_{video_name}.json'

        try:
            # Load ground truth annotations and predictions for the current video
            coco_gt = COCO(video_gt)
            coco_dt = coco_gt.loadRes(video_preds)

            # Open file to append results
            with open(per_class_results_path, "a") as per_class_file:
                per_class_file.write(f"\n=== Video: {video_name} ===\n")

                # Iterate over each class
                for cat_id in coco_gt.getCatIds():
                    cat_name = coco_gt.loadCats(cat_id)[0]["name"]
                    
                    # Filter ground truth and predictions by class
                    coco_gt_class = coco_gt.loadAnns(coco_gt.getAnnIds(catIds=[cat_id]))
                    coco_dt_class = coco_dt.loadAnns(coco_dt.getAnnIds(catIds=[cat_id]))

                    # Convert filtered annotations into COCO objects
                    if not coco_gt_class or not coco_dt_class:
                        per_class_file.write(f"\n--- Class: {cat_name} (ID: {cat_id}) ---\nNo detections for this class.\n")
                        continue  # Skip evaluation if no annotations

                    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
                    coco_eval.params.catIds = [cat_id]  # Filter evaluation by category
                    
                    # Run evaluation again for this class
                    coco_eval.evaluate()
                    coco_eval.accumulate()

                    # Redirect output to file
                    original_stdout = sys.stdout
                    sys.stdout = per_class_file
                    
                    per_class_file.write(f"\n--- Class: {cat_name} (ID: {cat_id}) ---\n")
                    coco_eval.summarize()
                    
                    sys.stdout = original_stdout  # Restore stdout

        except Exception as e:
            print(f"Error processing {video_name}: {e}")

    print(f"Per-class evaluation metrics saved to {per_class_results_path}")


def compute_mean_metrics_per_class(per_class_file, output_file="global_metrics.txt"):
    """
    Reads per-class COCO evaluation results, computes the mean for each metric per class, and saves the global results.

    :param per_class_file: Path to the .txt file containing per-class COCO metrics.
    :param output_file: Path to save the global per-class metrics results.
    :return: A dictionary with the mean values of each metric per class.
    """

    # Regular expressions
    class_pattern = re.compile(r'--- Class: (\w+)')  # Extracts class names (e.g., "car", "pedestrian")
    metric_pattern = re.compile(r'(.+?) = ([\d\.]+)')  # Extracts metric values

    metrics_per_class = {}  # Stores metrics for each class

    current_class = None  # Tracks the class currently being read

    with open(per_class_file, 'r') as file:
        for line in file:
            class_match = class_pattern.search(line)
            if class_match:
                current_class = class_match.group(1)  # Get class name
                if current_class not in metrics_per_class:
                    metrics_per_class[current_class] = {}  # Initialize storage for the class
            elif current_class:
                metric_match = metric_pattern.search(line)
                if metric_match:
                    metric_name = metric_match.group(1).strip()
                    metric_value = float(metric_match.group(2))

                    # Store metric values in a list for averaging later
                    if metric_name not in metrics_per_class[current_class]:
                        metrics_per_class[current_class][metric_name] = []
                    metrics_per_class[current_class][metric_name].append(metric_value)

    # Compute mean for each metric per class
    mean_metrics_per_class = {
        cls: {metric: np.mean(values) for metric, values in class_metrics.items()}
        for cls, class_metrics in metrics_per_class.items()
    }

    # Save results to a new file
    with open(output_file, "w") as out_file:
        out_file.write("Global COCO evaluation metrics per class (mean over all videos):\n\n")

        for class_name, class_metrics in mean_metrics_per_class.items():
            out_file.write(f"\n=== Class: {class_name} ===\n")
            for metric, mean_value in class_metrics.items():
                out_file.write(f"{metric}: {mean_value:.4f}\n")

    print(f"\nGlobal per-class mean metrics saved to {output_file}")

    return mean_metrics_per_class




if __name__ == '__main__':
    
    # EVALUATION PER VIDEO
    #per_video_evaluation()

    # EVALUATION PER VIDEO AND CLASS
    # per_class_evaluation()

    # MEAN EVALUATION PER CLASS ACROSS VIDEO
    # per_class_metrics_path = "/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/evaluation/per_class_metrics.txt"
    # compute_mean_metrics_per_class(per_class_metrics_path)

    # MEAN EVALUATION COMBINED CLASSES ACROSS
    file_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/evaluation_results/off-the-shelf/evaluation/per_video_metrics.txt'
    mean_metrics = compute_mean_metrics_from_txt(file_path)
