import os
import pickle
import argparse

import wandb
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor, PreTrainedTokenizer
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from evaluator import Evaluator
from task1 import model_forward
import utils


week4_path = "/ghome/c5mcv03/mcv-c5-group-3/week4/artifacts/fine-tune-both:v0/best_model.pth"
week5_path = "/ghome/c5mcv03/mcv-c5-group-3/week5/outputs/finetune_both_w5.pth"

# Code to download the artifact, if needed
# wandb.login(key='89f4c571fd157f9b9bd2d73a2e6c39eb0ed38ad2')
# run = wandb.init()
# artifact = run.use_artifact('arnalytics-universitat-aut-noma-de-barcelona/C5-W5/fine-tune-both:v0', type='model')
# artifact_dir = artifact.download("/ghome/c5mcv03/mcv-c5-group-3/week5/outputs")
GENERATION_KWARGS = {
    'max_length': 51,
}
BATCH_SIZE = 8

def test_loop(
        evaluator: Evaluator,
        model: VisionEncoderDecoderModel,
        test_dataloader: DataLoader,
        loss_fn: nn.Module,
        tokenizer: PreTrainedTokenizer,
        epoch: int,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        is_validation: bool = True,
        is_only_drinks: bool = False,
):
    model.eval()
    model.to(device)
    all_predictions = []
    all_ground_truth = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for img, text in tqdm(test_dataloader, desc="Evaluating"):
            img, text = img.to(device), text.to(device)

            # Compute loss using teacher forcing
            logits = model_forward(model, img, text, device)
            
            loss = loss_fn(logits, text[:, 1:].long())
            total_loss += loss.item() * img.size(0)
            num_samples += img.size(0)

            # Generate predictions instead of using argmax
            out = model.generate(img, **GENERATION_KWARGS)
            predictions = tokenizer.batch_decode(out, skip_special_tokens=True)
            ground_truth = tokenizer.batch_decode(text.long(), skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)

    test_loss = total_loss / num_samples if num_samples > 0 else 0.0
    metrics = evaluator.evaluate(all_ground_truth, all_predictions)

    stage = "validation" if is_validation else "test"
    string = ''
    if is_only_drinks:
        string = '_drinks'
    print(
        {f"{stage}_loss{string}": test_loss,
        **{f"{stage}_{k}{string}": v for k, v in metrics.items()}}
    )
    print(f"\n\n{stage.capitalize()}{string} Loss: {test_loss:.4f}")
    utils.pretty_print(metrics, stage.capitalize())

    return all_predictions, all_ground_truth




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image captioning models")
    parser.add_argument('--dataset_path', type=str, 
                        default="/ghome/c5mcv03/mcv-c5-group-3/week5/test_dataset_only_drinks.pkl",
                        help="Path to the test dataset pickle file")
    parser.add_argument('--output_filename', type=str,
                        default="fine-both.txt",
                        help="Output filename for the predictions and ground truth")
    args = parser.parse_args()
    
    output_filename = args.output_filename
    dataset_path = args.dataset_path
    
    # Create the test dataloader
    with open(dataset_path, "rb") as f:
        dataset_onlydrinks = pickle.load(f)
    test_dataloader = DataLoader(dataset_onlydrinks, batch_size=BATCH_SIZE, shuffle=False)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_NAME = 'nlpconnect/vit-gpt2-image-captioning'
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    evaluator = Evaluator()
    for model_path, output_folder in zip([week4_path, week5_path], ["week4", "week5"]):
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        model.load_state_dict(state_dict)
        loss_fn = nn.CrossEntropyLoss(ignore_index=model.config.decoder.pad_token_id)
        all_predictions, all_ground_truth = test_loop(
            evaluator,
            model,
            test_dataloader,
            loss_fn, tokenizer,
            epoch=9,
            device=device,
            is_validation=False,
            is_only_drinks=True
        )
        
        ## GENERATING QUALITATIVE OUTPUTS
        # Crear directorio para guardar resultados si no existe
        output_dir = f"/ghome/c5mcv03/mcv-c5-group-3/week5/outputs/{output_folder}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Ruta del archivo de salida
        output_file = os.path.join(output_dir, output_filename)
        
        # Abrir archivo para escribir
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PREDICCIONES VS GROUND TRUTH\n")
            f.write("="*50 + "\n\n")
            
            # Escribir cada predicción junto con su correspondiente ground truth
            for i in range(len(all_predictions)):
                f.write(f"Ejemplo #{i+1}:\n")
                f.write("-"*50 + "\n")
                f.write(f"PREDICCIÓN:\n{all_predictions[i]}\n\n")
                f.write(f"GROUND TRUTH:\n{all_ground_truth[i]}\n\n")
                f.write("="*50 + "\n\n")
            
        print(f"Resultados guardados en: {output_file}")
