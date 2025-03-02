from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# Cargar el ground truth
coco_gt = COCO("gt_coco_0000.json")

# Cargar las predicciones
coco_dt = coco_gt.loadRes("predictions_50.json")

# Crear el evaluador COCO
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

# for catId in coco_gt.getCatIds():
#     coco_eval.params.catIds = [catId]  # Evaluar solo esta clase
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#     print(f"Resultados para la categoría: {coco_gt.loadCats(catId)[0]['name']}")

# Evaluar y mostrar resultados
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()