from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

import consts
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional

class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        # Load model configuration and weights
        self.cfg.merge_from_file(model_zoo.get_config_file(consts.MODEL_NAME))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(consts.MODEL_NAME)

        # Set dataset for training and testing
        self.cfg.DATASETS.TRAIN = ("kitti_mots_training",)
        self.cfg.DATASETS.TEST = ("kitti_mots_training",)

        # Model settings
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Active only for evaluation
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # 0.5 for Inference
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(self.cfg)

        # self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        # self.metadata = MetadataCatalog.get("coco_2017_val")
        self.metadata = MetadataCatalog.get("kitti_mots_training")

    def run_inference(self, image: Image.Image) -> Dict:
        """
        Run object detection inference on a single image.
        """
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        outputs = self.predictor(image_cv)

        # 0 = Person, 2 = Car
        valid_classes = [0, 2]

        # Filter detections
        instances = outputs["instances"].to("cpu")
        keep = [i for i, cat in enumerate(instances.pred_classes) if cat.item() in valid_classes]

        # Keep valid detections
        outputs["instances"] = instances[keep]
        print(outputs)
        return outputs
    
    def visualize_detections(self, image: Image.Image,
                             results: Dict,
                             output_path: Optional[str] = None) -> Image.Image:
        """
        Visualize object detection results on an image.
        """
        image_cv = np.array(image)
        
        visualizer = Visualizer(image_cv, self.metadata, scale=0.8)
        output = visualizer.draw_instance_predictions(results["instances"].to("cpu"))
        output_image = Image.fromarray(output.get_image())
        
        if output_path:
            output_image.save(output_path)
            print(f"Saved detection result to {output_path}")

        return output_image
