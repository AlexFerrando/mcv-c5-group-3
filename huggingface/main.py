from pycocotools.mask import toBbox
import json
from coco_eval import CocoEvaluator, format_results_to_coco
from inference import load_model, run_inference
from read_data import read_data
import consts
from pycocotools.coco import COCO
from get_ground_truth import load_images_and_annotations_for_video


if __name__ == '__main__':
    # video0000_folder = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/training/image_02/0000'
    # annotations_file = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/instances_txt/0000.txt'
    video0000_folder = f'{consts.KITTI_MOTS_PATH_ALEX}training/image_02/0000'
    annotations_file = f'{consts.KITTI_MOTS_PATH_ALEX}instances_txt/0000.txt'
    output_json_path = './gt_coco_0000.json'

    gt_coco = load_images_and_annotations_for_video(video0000_folder, annotations_file)
    
    # Save the gt_coco as a JSON file
    #output_json_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/KITTI_MOTS/gt_coco_0000.json'
    with open(output_json_path, 'w') as f:
        json.dump(gt_coco, f, indent=4)
    print(f"Ground truth COCO annotations saved to: {output_json_path}")

    gt_coco = COCO(output_json_path)

    coco_evaluator = CocoEvaluator(gt_coco)

    # Load dataset
    dataset = read_data(consts.KITTI_MOTS_PATH_ALEX)
    dataset = dataset['train']['image']

    model, image_processor, device = load_model()
    results = []
    for i in range(0, len(dataset), 10):
        batch = dataset[i:min(i + 10, len(dataset))]
        results += run_inference(model, image_processor, batch, device)

    formatted_results = format_results_to_coco(results)

    coco_evaluator.update(formatted_results)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_results = coco_evaluator.prepare_for_coco_detection(formatted_results)
    
    import json
    with open('coco_results.json', 'w') as f:
        json.dump(coco_results, f, indent=4)
    print("COCO results saved to coco_results.json")

    