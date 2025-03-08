from ultralytics import YOLO
import os

def get_image_paths(sequence_path):
    return [os.path.join(sequence_path, f) for f in os.listdir(sequence_path) if f.endswith(".png")]


def main(model_name, sequence_path, save_path):
    model = YOLO(model_name) # load_model
    image_paths = get_image_paths(sequence_path)
    model.predict(image_paths)


if __name__ == "__main__":
    model_path = "yolo11n.pt"
    sequence_path = "/projects/master/c5/KITTI_MOTS/training/image_02/0000"
    save_path = "/projects/master/c5/results"
    main(model_path, sequence_path, save_path)
