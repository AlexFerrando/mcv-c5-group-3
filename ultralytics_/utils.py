import cv2
import matplotlib.pyplot as plt

def draw_yolo_bbox(image_path, annotation_path, save_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Convert image from BGR (OpenCV default) to RGB (for matplotlib)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = image.shape

    # Read the annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    # Process each line (each bounding box)
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            # Expecting 5 values: class, x_center, y_center, width, height
            continue

        cls, x_center, y_center, bbox_width, bbox_height = parts
        x_center, y_center = float(x_center), float(y_center)
        bbox_width, bbox_height = float(bbox_width), float(bbox_height)

        # Convert normalized coordinates to pixel values
        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        bbox_width_pixel = bbox_width * img_width
        bbox_height_pixel = bbox_height * img_height

        print(f"Line {i} - Class: {cls}, x_center (px): {x_center_pixel}, y_center (px): {y_center_pixel}, "
              f"width (px): {bbox_width_pixel}, height (px): {bbox_height_pixel}")


        # Calculate top-left and bottom-right coordinates
        x_min = int(x_center_pixel - bbox_width_pixel / 2)
        y_min = int(y_center_pixel - bbox_height_pixel / 2)
        x_max = int(x_center_pixel + bbox_width_pixel / 2)
        y_max = int(y_center_pixel + bbox_height_pixel / 2)
        print(f"Line {i} - Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")


        # Draw the bounding box and class label
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue rectangle
        cv2.putText(image, cls, (x_min, max(y_min - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    # Save the image with bounding boxes using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title("YOLO Annotations Visualization")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    annotation_file_path = "/ghome/c5mcv03/test/KITTI-MOTS-YOLO/0000_153.txt"  # Replace with your annotation file path
    image_file_path = "/ghome/c5mcv03/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/000153.png"  # Replace with your image file path
    save_image_path = "/ghome/c5mcv03/mcv-c5-group-3/ultralytics/test.png"  # Replace with your desired save path

    draw_yolo_bbox(image_file_path, annotation_file_path, save_image_path)
