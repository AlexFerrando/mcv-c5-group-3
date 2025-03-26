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
    wandb.log({
        f"{stage}_loss": test_loss,
        **{f"{stage}_{k}": v for k, v in metrics.items()}
    })
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
        wandb.log({"train_loss": train_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Validation phase
        val_loss = test_loop(evaluator, model, val_dataloader, loss_fn, tokenizer, device, is_validation=True)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch  
            best_model_state = model.state_dict() 
            print('BEST EPOCH SO FAR!')

    # Upload model to wandb
    if wandb.config['save_best_model'] and best_model_state is not None:
        
        # Load and save locally
        model_path = 'best_model.pth'
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, model_path)
        
        # Create artifact
        artifact = wandb.Artifact(
            name="best_model",
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
    
    test_loop(evaluator, model, test_dataloader, loss_fn, tokenizer, device, is_validation=False)
    

def setup_wandb(disabled: bool = False) -> None:
    # Experiment configuration
    config = {
        'batch_size': 5,
        'experiment': 'fine-tune-encoder',  # 'off-shelf', 'fine-tune-encoder', 'fine-tune-decoder', 'fine-tune-both'
        'lr': 1e-5,
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


if __name__ == '__main__':

    # Load tokenizer, feature extractor and model
    MODEL_NAME = 'nlpconnect/vit-gpt2-image-captioning'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Setup wandb
    setup_wandb(disabled=False)

    # Define dataset and create dataloaders
    dataset = FoodDataset(
        consts.DATA_PATH,
        tokenizer,
        feature_extractor,
        #transform=podem passar-li un augmentador d'imatges
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
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config['lr'])

    pipeline(
        wandb.config['experiment'],
        model,
        loss_fn,
        optimizer,
        train_dataloader, val_dataloader, test_dataloader,
        evaluator,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_epochs=wandb.config['num_epochs']
    )
