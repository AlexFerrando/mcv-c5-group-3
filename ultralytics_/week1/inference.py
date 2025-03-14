import consts

from ultralytics import YOLO
import os

def get_image_paths(sequence_path):
    return [os.path.join(sequence_path, f) for f in os.listdir(sequence_path) if f.endswith(".png")]


def main(model_name, sequence_path, save_path, inference_arguments):
    model = YOLO(model_name) # load_model
    image_paths = sorted(get_image_paths(sequence_path))
    print(image_paths)
    results = model.predict(image_paths[-350:], **inference_arguments)
    sequence_name = os.path.basename(sequence_path)
    for i, r in enumerate(results):
        print("saving image to", os.path.join(save_path, f"result_{sequence_name}_{int(i) + len(image_paths) - 350}.png"))
        r.save(os.path.join(save_path, f"result_{sequence_name}_{int(i) + len(image_paths) - 350}.png"))

if __name__ == "__main__":
    model_path = "yolo11n.pt"
    model_path = "/ghome/c5mcv03/mcv-c5-group-3/outputs/pol/models/yolo11n.pt"
    model_path = "/projects/master/c5/best.pt"
    
    sequence_path = "/ghome/c5mcv03/KITTI-MOTS-YOLO/validation/images"
    sequence_path = "/projects/master/c5/KITTI-MOTS-big/training/image_02/0019"
    

    output_inferred_images_path = os.path.join(consts.BASE_PATH_OUTPUT_LOCAL, "inference_images", "0019")
    os.makedirs(output_inferred_images_path, exist_ok=True)
    print(output_inferred_images_path)
    inference_arguments = {'classes': consts.YOLO_CLASSES, "stream": True}
    main(model_path, sequence_path, output_inferred_images_path, inference_arguments)
