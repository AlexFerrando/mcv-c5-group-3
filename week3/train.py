from dataset import FoodDataset
from our_tokenizers import CharacterTokenizer, BaseTokenizer
from metrics import Metric
import consts
from models import BaselineModel
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
        name=f"Baseline_{wandb.util.generate_id()}",
        config=config,
        reinit=True
    )
    
    # Log model architecture
    wandb.watch(model, log="all")
    
    # Set up criterion and metric
    criterion = nn.CrossEntropyLoss()
    metric = Metric()

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0  # Initialize patience counter
    
    for epoch in range(epochs):
        print('Epoch:', epoch)
        
        # Training step
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        print(f'train loss: {train_loss:.2f}')
        
        # Validation step
        val_loss, val_metrics = eval_epoch(model, criterion, val_loader, tokenizer, device, metric)
        print(f'valid loss: {val_loss:.2f}, metric: {val_metrics}')
        
        # Log metrics to wandb
        log_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        
        # Save best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter on improvement
            wandb.save("best_model.pth")
            log_data["best_val_loss"] = best_val_loss
        else:
            patience_counter += 1  # Increment patience counter if no improvement
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
        
        wandb.log(log_data)
        print('-------------------')

        # Check early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    # Final evaluation on the test set
    test_loss, test_metrics = eval_epoch(model, criterion, test_loader, tokenizer, device, metric)
    print(f'test loss: {test_loss:.2f}, metric: {test_metrics}')
    
    # Log test metrics
    wandb.log({
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()}
    })
    
    wandb.finish()


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:  
    losses = []
    model.train()
    for img, text in tqdm(dataloader, desc='Training epoch'):
        optimizer.zero_grad()
        img = img.to(device)
        text = text.to(device)
        out = model(img)
        loss = criterion(out, text.long())
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss


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
            }
        }
    }
    """
    # Initialize the tokenizer
    tokenizer = CharacterTokenizer()

    # Use the config for the transform (ensuring consistency)
    transform = nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(config['resize'], antialias=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    
    dataset = FoodDataset(
        data_path=consts.DATA_PATH_POL,
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
    
    model = BaselineModel(tokenizer=tokenizer)
    
    # Choose optimizer based on config without modifying it
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

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
        'learning_rate': 1e-4,
        'batch_size': 64,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'epochs': 300,
        'patience': 30,
        'project': 'C5-W3',
        'gradient_max_norm': 5.0
    }

    tokenizer = CharacterTokenizer()

    transform = nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(config['resize'], antialias=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    
    dataset = FoodDataset(
        data_path=consts.DATA_PATH_POL,
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
    
    model = BaselineModel(tokenizer=tokenizer)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_max_norm'])
    
    # Regular training call with explicit config parameters
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
    # wandb.agent(sweep_id, func
