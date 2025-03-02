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

        # Model settings
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(self.cfg)

        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        # self.metadata = MetadataCatalog.get("coco_2017_val") # COCO metadata
        # self.metadata = MetadataCatalog.get("kitti_mots_testing") # COCO metadata

    def run_inference(self, image: Image.Image) -> Dict:
        """
        Run object detection inference on a single image.
        """
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        outputs = self.predictor(image_cv)
        return outputs
    
    def visualize_detections(self, image: Image.Image,
                             results: Dict,
                             output_path: Optional[str] = None) -> Image.Image:
        """
        Visualize object detection results on an image.
        """
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        visualizer = Visualizer(image_cv[:, :, ::-1], self.metadata, scale=0.8)
        output = visualizer.draw_instance_predictions(results["instances"].to("cpu"))
        output_image = Image.fromarray(cv2.cvtColor(output.get_image(), cv2.COLOR_BGR2RGB))
        
        if output_path:
            output_image.save(output_path)
            print(f"Saved detection result to {output_path}")

        return output_image
