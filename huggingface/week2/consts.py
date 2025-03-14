# Path to data
from dataclasses import dataclass
from torch import Tensor


KITTI_MOTS_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS/'
KITTI_MOTS_PATH_ALEX = '/home/alex/Documents/MCV/C5/KITTI_MOTS/'

# Model info. Instance segmentation
MASK2FORMER = "facebook/mask2former-swin-small-coco-instance"
MAX_WIDTH_SIZE_KITTI = 1242
MAX_HEIGHT_SIZE_KITTI = 376
MAX_WIDTH_SIZE_CPPE5 = 480
MAX_HEIGHT_SIZE_CPPE5 = 480

INTERESTING_CLASSES = ['car', 'person'] 



# Classes mapping
def inverse_mapping_class_id(dataset: str=None, class_id: int=None):
    """Maps a class id from one dataset to the other.

    If dataset = 'coco' and class_id = 2 (class name = 'car') --> returns the id of the class 'car' in KITTI dataset.
    If dataset = 'kitti' and class_id = 1 (class name = 'car') --> returns the id of the class 'car' in COCO dataset.
    """
    
    if dataset == 'coco':
        if class_id == 3:  # 'car'
            return 1
        
        elif class_id == 1: # 'person'
            return 2
    
    elif dataset == 'kitti':
        if class_id == 1:  # 'car'
            return 3
        
        elif class_id == 2:  # 'person'
            return 1


# Classes mapping
KIITI_TO_COCO_IDS = {
    1: 1, # 'car'
    2: 1, # 'pedestrian'
}

@dataclass
class DetectionResults:
    scores: Tensor
    boxes: Tensor
    labels: Tensor