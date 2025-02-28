import consts
import torch
from PIL import ImageDraw
from read_data import read_data
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = read_data(consts.KITTI_MOTS_PATH_ALEX)
    image_processor = AutoImageProcessor.from_pretrained(consts.MODEL_NAME)
    model = AutoModelForObjectDetection.from_pretrained(consts.MODEL_NAME)
    model.to(device)
    
    image = dataset['test']['image'][0]
    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

        draw = ImageDraw.Draw(image)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1)
            draw.text((x, y), model.config.id2label[label.item()], fill="white")
        # Save the image
        image.save("output.jpg")
    

if __name__ == '__main__':
    main()  