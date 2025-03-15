import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO

def annotate_image_with_masks(result):
    orig_img = cv2.cvtColor(result[0].orig_img, cv2.COLOR_RGB2BGR)
    masks = result[0].masks

    img_overlay = orig_img.astype(np.float32) / 255.0
    alpha = 0.5

    for m in masks:
        if hasattr(m, 'cpu'):
            m = m.data.cpu().numpy()
        mask = m.squeeze()
        mask_binary = (mask > 0.5).astype(np.float32)
        color = np.random.rand(3)
        colored_mask = np.zeros_like(img_overlay)
        colored_mask[..., 0] = color[0]
        colored_mask[..., 1] = color[1]
        colored_mask[..., 2] = color[2]
        img_overlay = img_overlay * (1 - mask_binary[..., None] * alpha) + colored_mask * (mask_binary[..., None] * alpha)

    img_overlay = (img_overlay * 255).astype(np.uint8)
    return img_overlay

def main(model_name, folder_path):
    # Load the pretrained model
    model = YOLO(model_name)
    result = model.predict(folder_path, classes=[0, 2], show_boxes=False, retina_masks=True)

    # Example usage
    annotated_image = annotate_image_with_masks(result)

    # Display the resulting image with all masks overlaid
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.title("Image with Overlaid Masks")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate images with masks using YOLO model")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the YOLO model', default="yolo11n-seg.pt")
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images')

    args = parser.parse_args()
    main(args.model_name, args.folder_path)
