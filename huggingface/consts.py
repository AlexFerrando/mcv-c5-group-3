# Path to data
from dataclasses import dataclass
from torch import Tensor


KITTI_MOTS_PATH = '/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS/'
KITTI_MOTS_PATH_ALEX = '/home/alex/Documents/MCV/C5/KITTI_MOTS/'
KITTI_MOTS_PATH_RELATIVE = 'KITTI_MOTS/'

# Model info
MODEL_NAME = "facebook/detr-resnet-50"
MAX_WIDTH_SIZE = 1242
MAX_HEIGHT_SIZE = 376

@dataclass
class DetectionResults:
    scores: Tensor
    boxes: Tensor
    labels: Tensor