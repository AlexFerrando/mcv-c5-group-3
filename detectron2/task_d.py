from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog

from detector import Detector
import consts
from read_data import register_kitti_mots, load_kitti_mots_dataset

def evaluate_model(dataset_name, detector):
    """
    Evaluate the model using COCO metrics (AP, IoU) on the KITTI MOTS dataset.
    
    Args:
        dataset_name (str): Dataset name.
        detector (Detector): The initialized Detector instance.
    """  
    # Evaluate using COCO evaluator
    print("Starting model evaluation...")
    evaluator = COCOEvaluator(dataset_name, detector.cfg, False, output_dir="evaluation/")
    val_loader = build_detection_test_loader(detector.cfg, dataset_name)
    
    # Perform evaluation
    print("Running inference...")
    print(inference_on_dataset(detector.predictor.model, val_loader, evaluator))


if __name__ == '__main__':

    # Initialize Detector
    detector = Detector()

    dataset_name = "kitti_mots_training"
    if dataset_name in DatasetCatalog.list():
        print(f"Removing existing dataset '{dataset_name}' to avoid cache issues...")
        DatasetCatalog.remove(dataset_name)

    # Register dataset
    dataset_path = consts.KITTI_MOTS_PATH
    DatasetCatalog.register(dataset_name, lambda: load_kitti_mots_dataset(dataset_path))

    # Set metadata
    MetadataCatalog.get(dataset_name).set(thing_classes=["Person", "Car"])
    detector.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # Evaluate the model
    evaluate_model(dataset_name, detector)