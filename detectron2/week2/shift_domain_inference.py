import os
import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

# Define paths
image = '1_0'
MODEL_PATH = "./output/model_final.pth"  # Adjust if stored elsewhere
IMAGE_PATH = f'/ghome/c5mcv03/mcv-c5-group-3/detectron2/pannuke_images/fold{image}.jpg'
OUTPUT_PATH = f"./inference_output_{image}_2.jpg"

# Load configuration and model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Load and preprocess image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
outputs = predictor(image)

# Manually set CPPE-5 dataset metadata
cppe5_metadata = MetadataCatalog.get("pannuke_fold1")
cppe5_metadata.thing_classes = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

# Visualization
visualizer = Visualizer(image_rgb, metadata=cppe5_metadata, scale=1.0)
out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save output image
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR))
print(f"Inference complete! Output saved to {OUTPUT_PATH}")