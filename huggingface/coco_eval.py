from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load ground truth
coco_gt = COCO("gt_coco_0000.json")

# Load predictions
coco_dt = coco_gt.loadRes("results_coco_0000.json")

# Evaluate
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()