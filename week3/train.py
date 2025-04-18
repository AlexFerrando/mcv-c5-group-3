from dataset import FoodDataset
from our_tokenizers import CharacterTokenizer, BaseTokenizer
from metrics import Metric
import consts
from models import BaselineModel, LSTMModel, LSTMWithAttention
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision.transforms import v2
from tqdm import tqdm
import wandb
import os
import atexit


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: BaseTokenizer,
    epochs: int,
    patience: int,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    config: dict = None,
):
    # Initialize wandb with the complete config and (optional) project override
    wandb.init(
        entity="arnalytics-universitat-aut-noma-de-barcelona",
        project=config.get('project', 'C5-W3'),
        name=f"ATTENTION_{wandb.util.generate_id()}",
        config=config,
        reinit=True,
        # mode="offline" # Disable wandb logging for now
    )
    
    # Log model architecture
    wandb.watch(model, log="all")
    
    # Set up criterion and metric
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char2idx[tokenizer.pad_token])  # Ignore padding token
    metric = Metric()

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0  # Initialize patience counter
    
    for epoch in range(epochs):
        print('Epoch:', epoch)
        
        # Training step
        train_loss, train_metrics = train_epoch(
            model, 
            optimizer, 
            criterion, 
            train_loader, 
            tokenizer, 
            device, 
            metric,
            config
        )
        print(f'train loss: {train_loss:.2f}, metric: {train_metrics}')
        
        # Validation step
        val_loss, val_metrics = eval_epoch(model, criterion, val_loader, tokenizer, device, metric)
        print(f'valid loss: {val_loss:.2f}, metric: {val_metrics}')
        
        # Log metrics to wandb
        log_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()}
        }
        
        # Save best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter on improvement

            # Save model
            torch.save(model.state_dict(), config['model_name'])
            artifact = wandb.Artifact(name=config['model_name'].split('.')[0], type='model')  # Create artifact
            artifact.add_file(config['model_name'])
            wandb.log_artifact(artifact)
            log_data["best_val_loss"] = best_val_loss
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
        
        wandb.log(log_data)
        print('-------------------')

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    # Final evaluation on the test set
    test_loss, test_metrics = eval_epoch(model, criterion, test_loader, tokenizer, device, metric)
    print(f'test loss: {test_loss:.2f}, metric: {test_metrics}')
    
    wandb.log({
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()}
    })
    
    wandb.finish()


def get_grad_norm(model):
    return sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: BaseTokenizer,
    device: torch.device,
    metric: Metric,
    config: dict
) -> tuple:  
    all_texts = []
    all_texts_gt = []
    losses = []
    model.train()
    
    use_grad_clipping = config.get('use_grad_clipping', True)  # <--- toggle from config
    max_norm = config.get('gradient_max_norm', 5.0)
    use_teacher_forcing = config.get('use_teacher_forcing', True)
    detach_loop = config.get('detach_loop', False)

    for img, text in tqdm(dataloader, desc='Training epoch'):
        optimizer.zero_grad()
        img = img.to(device)
        text = text.to(device)
        if use_teacher_forcing:
            out = model(img, target_seq=text, teacher_forcing=True, detach_loop=detach_loop)
        else:
            out = model(img, teacher_forcing=False, detach_loop=detach_loop)

        loss = criterion(out, text.long())
        loss.backward()

        # --- Conditionally clip or just measure gradient norm ---
        if use_grad_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        else:
            grad_norm = get_grad_norm(model)
        wandb.log({"grad_norm": grad_norm}, commit=False)
        
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        
        texts = model.logits_to_text(out)
        all_texts.extend(texts)
        all_texts_gt.extend([tokenizer.decode(t.tolist()) for t in text])
    
    mean_loss = sum(losses) / len(losses)
    res = metric.compute_metrics(all_texts_gt, all_texts)
    return mean_loss, res


def eval_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: BaseTokenizer,
    device: torch.device,
    metric: Metric
) -> tuple:
    all_texts = []
    all_texts_gt = []
    losses = []
    model.eval()
    with torch.no_grad():
        for img, text in tqdm(dataloader, desc='Evaluation step'):
            img = img.to(device)
            text = text.to(device)
            out = model(img)
            loss = criterion(out, text.long())
            losses.append(loss.detach().cpu().item())
            texts = model.logits_to_text(out)
            all_texts.extend(texts)
            all_texts_gt.extend([tokenizer.decode(t.tolist()) for t in text])
    mean_loss = sum(losses) / len(losses)
    res = metric.compute_metrics(all_texts_gt, all_texts)
    return mean_loss, res


def get_split_sizes(dataset_len, train_size, val_size, test_size):
    assert train_size + val_size + test_size == 1, 'The sum of the sizes must be 1'
    train_size = int(train_size * dataset_len)
    val_size = (dataset_len - train_size) // 2
    test_size = dataset_len - train_size - val_size
    return train_size, val_size, test_size


def get_optimizer(config, model):
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'sgd':
        return torch.optim.SGD(
            model.parameters(), 
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def sweep_train(config: dict):
    """
    Example sweep config:

    config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-3, 1e-4, 5e-4]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'weight_decay': {
                'values': [0, 0.01, 0.001]
            },
            'epochs': {
                'value': 300
            },
            'patience': {
                'value': 300
            },
            'resize': {
                'value': (224, 224)
            },
            'gradient_max_norm': {
                'value': 5.0
            },
            'use_grad_clipping': {
                'values': [True, False]
            },
            'model_name': {
                'value': 'baseline.pth'
            },
            'resnet_model': {
                'value': 'microsoft/resnet-18'
            },
            'use_teacher_forcing': {          # <--- added teacher forcing option for sweeps
                'values': [True, False]
            }
        }
    }
    """
    # Initialize the tokenizer
    tokenizer = CharacterTokenizer()
    transform = nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(config['resize'], antialias=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    
    dataset = FoodDataset(
        data_path=consts.DATA_PATH,
        tokenizer=tokenizer,
        transform=transform
    )
    
    train_size, val_size, test_size = get_split_sizes(len(dataset), 0.8, 0.1, 0.1)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(consts.SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = LSTMModel(tokenizer=tokenizer, resnet_model=config['resnet_model'], lstm_layers=1)
    
    # Choose optimizer based on config without modifying it
    optimizer = get_optimizer(config, model)

    # Call train with explicit parameters from config
    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        epochs=config['epochs'],
        patience=config['patience'],
        config=config
    )


def cleanup_wandb():
    os.environ.pop('WANDB_API_KEY', None)
    wandb.finish()


if __name__ == '__main__':
    wandb.login(key='89f4c571fd157f9b9bd2d73a2e6c39eb0ed38ad2')

    config = {
        'resize': (224, 224),
        'lstm_layers': 1,
        'learning_rate': 1e-4,
        'batch_size': 2,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'epochs': 20,
        'patience': 5,
        'project': 'C5-W3',
        'gradient_max_norm': 5.0,
        'use_grad_clipping': True,
        'model_name': 'baseline.pth',
        'resnet_model': 'microsoft/resnet-34',
        'use_teacher_forcing': True,
        'detach_loop': False
    }

    tokenizer = CharacterTokenizer()

    transform = nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(config['resize'], antialias=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    dataset = FoodDataset(
        data_path=consts.DATA_PATH,
        tokenizer=tokenizer,
        transform=transform
    )   

    # Split dataset: 80% train, 10% val, 10% test
    train_size, val_size, test_size = get_split_sizes(len(dataset), 0.8, 0.1, 0.1)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(consts.SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = LSTMWithAttention(tokenizer=tokenizer, resnet_model=config['resnet_model'], lstm_layers=config['lstm_layers'])
    
    optimizer = get_optimizer(config, model)

    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        epochs=config['epochs'],
        patience=config['patience'],
        config=config
    )

    atexit.register(cleanup_wandb)    
    # Uncomment below to run a sweep (ensure you have set up your sweep config accordingly)
    # sweep_id = wandb.sweep(sweep_config, project="food-recognition-sweeps")
    # wandb.agent(sweep_id, func=sweep_train)
