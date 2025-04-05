from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor, PreTrainedTokenizer
from dataset import FoodDataset
from evaluator import Evaluator
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import consts
import torch
import wandb
import utils
import os
import numpy as np

# Check parameters here: https://huggingface.co/docs/transformers/main_classes/text_generation
GENERATION_KWARGS = {
    'max_length': 51,
}

def model_forward(model: VisionEncoderDecoderModel, img: torch.Tensor, text: torch.Tensor, device: torch.device) -> torch.Tensor:
    decoder_input_ids = text[:, :-1].long()
    return model.forward(
        img,
        decoder_input_ids=decoder_input_ids,
    ).logits.permute(0, 2, 1)


def test_loop(
        evaluator: Evaluator,
        model: VisionEncoderDecoderModel,
        test_dataloader: DataLoader,
        loss_fn: nn.Module,
        tokenizer: PreTrainedTokenizer,
        epoch: int,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        is_validation: bool = True
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
    wandb.log(
        {f"{stage}_loss": test_loss,
        **{f"{stage}_{k}": v for k, v in metrics.items()}}, step=epoch+1
    )
    print(f"\n\n{stage.capitalize()} Loss: {test_loss:.4f}")
    utils.pretty_print(metrics, stage.capitalize())

    return test_loss


def train_loop(
        model: VisionEncoderDecoderModel,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        evaluator: Evaluator,
        tokenizer: PreTrainedTokenizer,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_epochs: int = 10
):
    model.to(device)
    
    # Placeholders for best model
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_samples = 0

        for img, text in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            img, text = img.to(device), text.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model_forward(model, img, text, device)
            
            # Compute loss
            loss = loss_fn(logits, text[:, 1:].long())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * img.size(0)
            num_samples += img.size(0)
        
        train_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        # Validation phase
        val_loss = test_loop(evaluator, model, val_dataloader, loss_fn, tokenizer, epoch, device, is_validation=True)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch  
            best_model_state = model.state_dict() 
            print('BEST EPOCH SO FAR!')

        wandb.log({"train_loss": train_loss, "epoch": epoch + 1}, step=epoch+1)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Upload model to wandb
    if wandb.config['save_best_model'] and best_model_state is not None:
        
        # Load and save locally
        model_path = 'best_model.pth'
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, model_path)
        
        # Create artifact
        artifact = wandb.Artifact(
            name=wandb.config['experiment'],
            type="model",
            metadata={
                'loss': best_val_loss,
                'epoch': best_epoch,
                'architecture': model.__class__.__name__
            }
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        os.remove(model_path)


def pipeline(
        experiment: str,
        model: VisionEncoderDecoderModel,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        evaluator: Evaluator,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_epochs: int = 10
):  
    utils.unfreeze(model)
    if experiment == 'off-shelf':
        utils.freeze(model)
    elif experiment == 'fine-tune-encoder':
        utils.freeze(model.decoder)
        train_loop(model, loss_fn, optimizer, train_dataloader, val_dataloader, evaluator, tokenizer, device=device, num_epochs=num_epochs)
    elif experiment == 'fine-tune-decoder':
        utils.freeze(model.encoder)
        train_loop(model, loss_fn, optimizer, train_dataloader, val_dataloader, evaluator, tokenizer, device=device, num_epochs=num_epochs)
    elif experiment == 'fine-tune-both':
        train_loop(model, loss_fn, optimizer, train_dataloader, val_dataloader, evaluator, tokenizer, device=device, num_epochs=num_epochs)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    # Final test evaluation
    test_loop(evaluator, model, test_dataloader, loss_fn, tokenizer, num_epochs-1, device, is_validation=False)
    

def setup_wandb(disabled: bool = False) -> None:
    # Experiment configuration
    config = {
        'batch_size': 5,
        'experiment': 'fine-tune-encoder',  # 'off-shelf', 'fine-tune-encoder', 'fine-tune-decoder', 'fine-tune-both'
        'lr': 5e-4,
        'num_epochs': 10,
        'save_best_model': True
    }
    
    wandb.login(key='89f4c571fd157f9b9bd2d73a2e6c39eb0ed38ad2')

    wandb.init(
        entity="arnalytics-universitat-aut-noma-de-barcelona",
        project='C5-W4',
        name=f"{config['experiment']}_{wandb.util.generate_id()}",
        config=config,
        reinit=True,
        mode='disabled' if disabled else 'online'
    )

def get_layerwise_lr_params(model, base_lr, lr_scale=1000.0):
    # Get all modules in forward order
    layers = list(model.named_parameters())
    num_layers = len(layers)
    
    # Exponential scaling: from base_lr / lr_scale to base_lr
    lrs = np.logspace(-np.log10(lr_scale), 0, num=num_layers) * base_lr

    param_groups = []
    for (name, param), lr in zip(layers, lrs):
        param_groups.append({'params': [param], 'lr': lr})
    return param_groups



if __name__ == '__main__':
    model_name = 'fine-tune-decoder:v0'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/ghome/c5mcv03/mcv-c5-group-3/week4/artifacts/' + model_name + '/best_model.pth'
    
    # Load tokenizer, feature extractor and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = ViTImageProcessor.from_pretrained(model_path)
    model = torch.load_state_dict(model_path)

    # Define dataset and create dataloaders
    dataset = FoodDataset(
        consts.DATA_PATH,
        tokenizer,
        feature_extractor,
    )

    # Split in train, validation and test
    train_size, val_size, test_size = utils.get_split_sizes(len(dataset), 0.8, 0.1, 0.1)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(consts.SEED)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)

    # Evaluate on test set
    evaluator = Evaluator()
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.config.decoder.pad_token_id)

    optimizer = optim.AdamW(get_layerwise_lr_params(model, base_lr=wandb.config['lr'], lr_scale=1000.0))

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


    # Crear directorio para guardar resultados si no existe
    output_dir = "prediction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Ruta del archivo de salida
    output_file = os.path.join(output_dir, "fine-tune_encoder.txt")
    
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
