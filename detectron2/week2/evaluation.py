from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

from detector import Detector
import consts
from read_data_help import CustomKittiMotsDataset

def evaluate_model(dataset_name, detector):
    """
    Evaluate the model using COCO metrics (AP, IoU) on the KITTI MOTS dataset.
    
    Args:
        dataset_name (str): Dataset name.
        detector (Detector): The initialized Detector instance.
    """  
    # Create predictor
    predictor = DefaultPredictor(detector.cfg)

    # Evaluate using COCO evaluator
    print("Starting model evaluation...")
    evaluator = COCOEvaluator(dataset_name, detector.cfg, False, output_dir="evaluation/")
    val_loader = build_detection_test_loader(detector.cfg, dataset_name)
    
    # Perform evaluation
    print("Running inference...")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == '__main__':

    detector = Detector()

    # Register dataset
    dataset_name = "kitti_mots"
    DatasetCatalog.register(dataset_name, lambda: CustomKittiMotsDataset(consts.KITTI_MOTS_PATH, use_coco_ids=True, split="val"))

    # Set metadata
    coco_classes = [""] * 81
    coco_classes[0] = "person"
    coco_classes[2] = "car"
    MetadataCatalog.get(dataset_name).set(thing_classes=coco_classes)

    # Evaluate the model
    evaluate_model(dataset_name, detector)