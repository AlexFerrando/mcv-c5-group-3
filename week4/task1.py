from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from dataset import FoodDataset
from evaluator import Evaluator
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import consts
import torch
import wandb
import utils


# Check parameters here: https://huggingface.co/docs/transformers/main_classes/text_generation
GENERATION_KWARGS = {
    'max_length': 201,
}

def evaluate_on_test(
        evaluator: Evaluator,
        model: VisionEncoderDecoderModel,
        test_dataloader: DataLoader,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    model.eval()
    model.to(device)
    all_predictions = []
    all_ground_truth = []
    with torch.no_grad():
        for img, text in tqdm(test_dataloader, desc='Testing'):
            out = model.generate(img.to(device), **GENERATION_KWARGS)
            predictions = tokenizer.batch_decode(out, skip_special_tokens=True)
            ground_truth = tokenizer.batch_decode(text.to(torch.long), skip_special_tokens=True)
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)
    metrics = evaluator.evaluate(all_ground_truth, all_predictions)
    wandb.log({
        **{f"test_{k}": v for k, v in metrics.items()}
    })
    utils.pretty_print(metrics, 'Test')

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
        pass
    elif experiment == 'fine-tune-encoder':
        raise NotImplementedError('Fine-tune encoder not implemented')
    elif experiment == 'fine-tune-decoder':
        raise NotImplementedError('Fine-tune decoder not implemented')
    elif experiment == 'fine-tune-both':
        raise NotImplementedError('Fine-tune both not implemented')
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    evaluate_on_test(evaluator, model, test_dataloader, device=device)
    

def setup_wandb(disabled: bool = False) -> None:
    wandb.login(key='89f4c571fd157f9b9bd2d73a2e6c39eb0ed38ad2')
    config = {
        'batch_size': 2,
        'experiment': 'off-shelf'
    }
    wandb.init(
        entity="arnalytics-universitat-aut-noma-de-barcelona",
        project='C5-W4',
        name=f"OFF_SHELF_{wandb.util.generate_id()}",
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
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.all_special_ids)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    pipeline(
        'off-shelf',
        model,
        loss_fn,
        optimizer,
        train_dataloader, val_dataloader, test_dataloader,
        evaluator,
        device=torch.device('cpu'),#torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_epochs=10
    )
