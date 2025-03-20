from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


import os
import re
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def evaluate_video_instance_segmentation(video_id, gt_path, pred_path):
    """
    Evalúa la segmentación de instancias para un video utilizando métricas COCO
    con mejor manejo de errores
    """
    # Cargar archivos
    gt_file = os.path.join(gt_path, f"gt_coco_{video_id}.json")
    pred_file = os.path.join(pred_path, f"preds_coco_{video_id}.json")
    
    # Verificar que los archivos existen
    if not os.path.exists(gt_file):
        return f"Error: El archivo GT {gt_file} no existe"
    if not os.path.exists(pred_file):
        return f"Error: El archivo de predicciones {pred_file} no existe"
    
    try:
        # Cargar GT
        coco_gt = COCO(gt_file)
        
        # Verificar que hay anotaciones en GT
        if len(coco_gt.getAnnIds()) == 0:
            return "Error: No hay anotaciones en el archivo GT"
        
        # Cargar predicciones
        with open(pred_file, 'r') as f:
            preds = json.load(f)
        
        # Verificar que hay predicciones
        if not preds or len(preds) == 0:
            return "Error: No hay predicciones en el archivo de predicciones"
        
        # Intentar cargar resultados con manejo de excepciones
        try:
            coco_dt = coco_gt.loadRes(preds)
        except IndexError:
            return "Error: Problema al cargar predicciones. Verificar formato de predicciones"
        
        # Diccionario para resultados
        results = {'overall': {}, 'by_class': {}}
        
        # Evaluación general
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        results['overall'] = {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2]
        }
        
        # Evaluación por clase - obtener categorías con manejo de excepciones
        try:
            categories = coco_gt.getCatIds()
            if not categories:
                return "Error: No hay categorías definidas en el archivo GT"
        except IndexError:
            return "Error: Problema al obtener categorías. Verificar formato del archivo GT"
        
        # Evaluación por clase
        for cat_id in categories:
            try:
                # Obtener nombre de la categoría
                category_info = coco_gt.loadCats(cat_id)[0]
                category_name = category_info.get('name', f"Categoría {cat_id}")
                
                # Evaluación para esta clase
                class_eval = COCOeval(coco_gt, coco_dt, 'segm')
                class_eval.params.catIds = [cat_id]
                class_eval.evaluate()
                class_eval.accumulate()
                class_eval.summarize()
                
                # Guardar resultados
                results['by_class'][cat_id] = {
                    'name': category_name,
                    'AP': class_eval.stats[0],
                    'AP50': class_eval.stats[1],
                    'AP75': class_eval.stats[2]
                }
            except Exception as e:
                results['by_class'][cat_id] = {
                    'name': f"Categoría {cat_id}",
                    'error': str(e)
                }
        
        return results
        
    except Exception as e:
        return f"Error evaluando video {video_id}: {str(e)}"

def evaluate_all_videos(gt_path: str, pred_path: str, output_file: str):
    """
    Evalúa todos los videos y genera un informe en formato .txt
    
    Args:
        gt_path (str): Ruta al directorio con archivos ground truth
        pred_path (str): Ruta al directorio con archivos de predicciones
        output_file (str): Ruta del archivo de salida .txt
    """
    # Encontrar todos los IDs de video disponibles
    video_ids = set()
    
    # Extraer IDs de los archivos GT
    for filename in os.listdir(gt_path):
        if filename.startswith('gt_coco_'):
            match = re.search(r'gt_coco_(\d+)\.json', filename)
            if match:
                video_ids.add(match.group(1))
    
    # Verificar que los archivos de predicciones también existen
    video_ids = [vid_id for vid_id in video_ids 
                if os.path.exists(os.path.join(pred_path, f"preds_coco_{vid_id}.json"))]
    
    # Ordenar IDs para procesar sistemáticamente
    video_ids = sorted(video_ids)
    
    # Evaluar cada video y escribir resultados
    with open(output_file, 'w') as f:
        f.write("# Evaluación de Instance Segmentation por Video y Clase\n\n")
        
        # Variables para métricas promedio
        all_metrics = {}
        
        for video_id in video_ids:
            f.write(f"## Video {video_id}\n\n")
            
            # Evaluar este video
            try:
                results = evaluate_video_instance_segmentation(video_id, gt_path, pred_path)
                
                # Escribir métricas generales
                f.write("### Métricas generales\n")
                f.write(f"AP: {results['overall']['AP']:.4f}\n")
                f.write(f"AP50: {results['overall']['AP50']:.4f}\n")
                f.write(f"AP75: {results['overall']['AP75']:.4f}\n\n")
                
                # Escribir métricas por clase
                f.write("### Métricas por clase\n")
                for cat_id, metrics in results['by_class'].items():
                    class_name = metrics['name']
                    f.write(f"Clase {cat_id} ({class_name}):\n")
                    f.write(f"  AP: {metrics['AP']:.4f}\n")
                    f.write(f"  AP50: {metrics['AP50']:.4f}\n")
                    f.write(f"  AP75: {metrics['AP75']:.4f}\n\n")
                    
                    # Acumular métricas para promedios
                    if class_name not in all_metrics:
                        all_metrics[class_name] = {'AP': [], 'AP50': [], 'AP75': []}
                    all_metrics[class_name]['AP'].append(metrics['AP'])
                    all_metrics[class_name]['AP50'].append(metrics['AP50'])
                    all_metrics[class_name]['AP75'].append(metrics['AP75'])
                
            except Exception as e:
                f.write(f"Error evaluando video {video_id}: {str(e)}\n\n")
        
        # Escribir promedio de métricas por clase
        f.write("## Promedios por clase en todos los videos\n\n")
        for class_name, metrics in all_metrics.items():
            f.write(f"Clase: {class_name}\n")
            f.write(f"  AP promedio: {sum(metrics['AP'])/len(metrics['AP']):.4f}\n")
            f.write(f"  AP50 promedio: {sum(metrics['AP50'])/len(metrics['AP50']):.4f}\n")
            f.write(f"  AP75 promedio: {sum(metrics['AP75'])/len(metrics['AP75']):.4f}\n\n")
    
    print(f"Evaluación completada. Resultados guardados en {output_file}")

if __name__ == '__main__':
    gt_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/week2/Evaluation_off-the-shelf/ground_truth'
    pred_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/huggingface/week2/Evaluation_off-the-shelf/preds_off-the-shelf'
    output_file = 'instance_segmentation_results.txt'

    evaluate_all_videos(gt_path, pred_path, output_file)