from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from tqdm import tqdm
import sys


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



if __name__ == '__main__':
    per_video_evaluation()
    
