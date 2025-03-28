from transformers import Blip2ForConditionalGeneration, AutoProcessor
from dataset_task2 import FoodDataset
from evaluator import Evaluator
from tqdm import tqdm
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split
import consts
import torch
import wandb
import utils
import os

GENERATION_KWARGS = {
    'max_new_tokens': 50,
    'do_sample': False,
    'temperature': 1.0,
}


def evaluate_blip2(model, dataloader, processor, evaluator, device, stage="test"):
    model.eval()
    model.to(device)
    all_predictions = []
    all_ground_truth = []

    logged = False
    with torch.no_grad():
        for images, titles in tqdm(dataloader, desc=f"Evaluating [{stage}]"):
            images = images.to(device)

            # prompts = ["Describe the food."] * len(images)
            inputs = processor(images=list(images), return_tensors="pt", padding=True).to(device)
            outputs = model.generate(**inputs, **GENERATION_KWARGS)
            predictions = processor.batch_decode(outputs, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_ground_truth.extend([t for t in titles])
            
            # Logging some images for easy qualitative analyisis
            if not logged:
                images_to_log = []
                print("images shape", images.shape)
                for img, pred, gt in zip(images, predictions, titles):
                    print("single image shape", img.shape)
                    if isinstance(img, torch.Tensor):
                        if img.ndim == 3 and img.shape[-1] == 3:
                            img = img.permute(2, 0, 1)
                        print("single image cpu", img.cpu().shape)
                        img = T.ToPILImage()(img.cpu())
                    images_to_log.append(wandb.Image(img, caption=f"GT: {gt}\nPred: {pred}"))
                wandb.log({f"{stage}_samples": images_to_log})
                logged = True

    metrics = evaluator.evaluate(all_ground_truth, all_predictions)
    wandb.log({f"{stage}_{k}": v for k, v in metrics.items()})
    print(f"\n\n{stage.capitalize()} Results:")
    utils.pretty_print(metrics, stage.capitalize())


def setup_wandb(disabled: bool = False):
    config = {
        'batch_size': 8,
        'experiment': 'blip2-eval',
        'num_epochs': 1,
        'save_best_model': False
    }

    wandb.login(key='8410a2da3f323633210ca8d25ce6862368d9f489')
    wandb.init(
        entity="arnalytics-universitat-aut-noma-de-barcelona",
        project='C5-W4',
        name=f"{config['experiment']}_{wandb.util.generate_id()}",
        config=config,
        reinit=True,
        mode='disabled' if disabled else 'online'
    )


if __name__ == '__main__':
    MODEL_NAME = 'Salesforce/blip2-opt-2.7b'

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    setup_wandb(disabled=False)

    dataset = FoodDataset(
        consts.DATA_PATH,
        tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
        transform=None
    )

    train_size, val_size, test_size = utils.get_split_sizes(len(dataset), 0.8, 0.1, 0.1)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(consts.SEED)
    )

    val_dataloader = DataLoader(val_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)

    evaluator = Evaluator()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate_blip2(model, val_dataloader, processor, evaluator, device, stage="validation")
    evaluate_blip2(model, test_dataloader, processor, evaluator, device, stage="test")
