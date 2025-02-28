from detector import *
from read_data import read_data

def main():
    # Load model
    detector = Detector()

    # Load data
    dataset = read_data(consts.KITTI_MOTS_PATH_RELATIVE)
    image = dataset['test']['image'][0]

    # Run inference
    results = detector.run_inference(image)

    # Visualize and save
    output_path = "output.jpg"
    detector.visualize_detections(image, results, output_path)
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    main()