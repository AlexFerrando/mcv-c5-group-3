from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import (
    VisionEncoderDecoderModel,
    ViTModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    ViTConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset import FoodDataset
import albumentations as A
from tqdm import tqdm
import numpy as np
import wandb

from evaluator import Evaluator
import utils
import consts

# -----------------------------------------------------------------------------
# CONFIGURATION (all options are in one place)
# -----------------------------------------------------------------------------
CONFIG = {
    'llama_model': 'meta-llama/Llama-3.2-1B', # meta-llama/Llama-3.2-3B
    'encoder_decoder_model': 'nlpconnect/vit-gpt2-image-captioning',
    'vit_encoder_path': None,  # Optionally, provide a local path to load a ViT encoder
    'batch_size': 4,
    'lr': 1e-4,
    'num_epochs': 5,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
    'max_seq_length': 128,
    'save_best_model': True,
    'experiment_type': 'peft_lora_finetune_simplified',
    'generation_kwargs': {'max_new_tokens': 50, 'num_beams': 3},
    'data_path': consts.DATA_PATH,
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def setup_wandb(config):
    wandb.init(project="C5-W4-PEFT", config=config, reinit=True, mode='disabled')
    print("Wandb initialized.")

def load_models(config, device):
    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(config['llama_model'])
    feature_extractor = ViTImageProcessor.from_pretrained(config['encoder_decoder_model'])

    vit_encoder = ViTModel(ViTConfig())
    # Load ViT encoder (from a local path if provided)
    if config.get('vit_encoder_path'):
        local_model = torch.load(config['vit_encoder_path'])
        encoder_only_model = OrderedDict(
            (k.replace("encoder.", "", 1), v)
            for k, v in local_model.items()
            if k.startswith('encoder')
        )

        vit_encoder.load_state_dict(encoder_only_model)
        print("ViT encoder loaded from local path.")
    
    # Load Llama decoder and apply PEFT (LoRA)
    decoder = AutoModelForCausalLM.from_pretrained(config['llama_model'])
    decoder.config.is_decoder = True

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none"
    )
    decoder = get_peft_model(decoder, peft_config)
    
    # Create the VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel(enoder=vit_encoder, decoder=decoder)

    # Freeze encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    # Since we are using Llama and it does not have pad token we need to set the pad_token_id to eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    model.to(device)
    return model, tokenizer, feature_extractor

def get_transforms():
    transform = A.Compose([
        A.NoOp()]
    )
    return transform

def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images, labels=labels, return_dict=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, device, evaluator, tokenizer, generation_kwargs):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_ground_truth = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images, labels=labels, return_dict=True)
            loss = outputs.loss
            total_loss += loss.item() * images.size(0)
            
            # Generate predictions using the model's generate method
            preds = model.generate(pixel_values=images, **generation_kwargs)
            
            # Decode both predictions and ground truth texts
            preds_decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
            gt_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(preds_decoded)
            all_ground_truth.extend(gt_decoded)
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Compute evaluation metrics comparing predictions with ground truth
    metrics = evaluator.evaluate(all_ground_truth, all_predictions)
    print("Validation Metrics:", metrics)
    
    return avg_loss, metrics

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_wandb(CONFIG)

    model, tokenizer, feature_extractor = load_models(CONFIG, device)

    # No transformations right now
    transform = get_transforms()

    # Prepare dataset and split (80/20 train/validation split)
    dataset = FoodDataset(CONFIG['data_path'], tokenizer, feature_extractor, CONFIG['max_seq_length'],
                          transform=transform)
    train_size, val_size, test_size = utils.get_split_sizes(len(dataset), 0.8, 0.1, 0.1)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(consts.SEED)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'],
                                  shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=wandb.config['batch_size'],
                                shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=wandb.config['batch_size'],
                                 shuffle=False, collate_fn=collate_fn)


    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    
    best_val_loss = float('inf')
    best_model_path = "/ghome/c5mcv03/mcv-c5-group-3/archive/artifacts/fine-tune-both:v0/best_model.pth"
    evaluator = Evaluator()

    for epoch in range(CONFIG['num_epochs']):
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_dataloader, device, evaluator, tokenizer, CONFIG['generation_kwargs'])
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save only the best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
    test_loss = validate_epoch(model, test_dataloader, device)
    wandb.log({"test_loss": test_loss})

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    main()
