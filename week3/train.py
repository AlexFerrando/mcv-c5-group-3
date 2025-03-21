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


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: BaseTokenizer,
    epochs: int = 10,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    # Criterion an metric
    criterion = nn.CrossEntropyLoss()
    metric = Metric()

    model.to(device)
    for epoch in range(epochs):
        print('Epoch:', epoch)
        loss, res = train_epoch(model, optimizer, criterion, metric, train_loader, device, tokenizer)
        print(f'train loss: {loss:.2f}, metric: {res:.2f}, epoch: {epoch}')
        loss_v, res_v = eval_epoch(model, criterion, metric, val_loader, device, tokenizer)
        print(f'valid loss: {loss_v:.2f}, metric: {res_v:.2f}')
        print('-------------------')
    loss_t, res_t = eval_epoch(model, criterion, metric, test_loader, device)
    print(f'test loss: {loss_t:.2f}, metric: {res_t:.2f}')
    
def train_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        metric: Metric,
        dataloader: DataLoader,
        device: torch.device,
        tokenizer: BaseTokenizer
):  
    all_outs = []
    all_texts = []
    losses = []
    model.train()

    for img, text in tqdm(dataloader, desc='Training epoch'):
        optimizer.zero_grad()
        img: torch.Tensor = img.to(device)
        text: torch.Tensor = text.to(device)
        out: torch.Tensor = model(img)
        loss: torch.Tensor = criterion(out, text.long())
        loss.backward()
        optimizer.step()

        # Save the outputs and texts for the metric
        all_outs.append(out.detach().cpu())
        all_texts.append(text.detach().cpu())
        losses.append(loss.item())
    out = torch.cat(all_outs, dim=0)
    text = torch.cat(all_texts, dim=0)
    mean_loss = sum(losses) / len(losses)
    res = metric.compute_metrics(out, text)
    return mean_loss, res


def eval_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    metric: Metric,
    dataloader: DataLoader,
    device: torch.device
):
    all_outs = []
    all_texts = []
    losses = []
    model.eval()
    for img, text in tqdm(dataloader, desc='Evaluation step'):
        img: torch.Tensor = img.to(device)
        text: torch.Tensor = text.to(device)
        out: torch.Tensor = model(img)
        loss: torch.Tensor = criterion(out, text.long())

        # Save the outputs and texts for the metric
        all_outs.append(out.detach().cpu())
        all_texts.append(text.detach().cpu())
        losses.append(loss.item())
    out = torch.cat(all_outs, dim=0)
    text = torch.cat(all_texts, dim=0)
    mean_loss = sum(losses) / len(losses)
    res = metric.compute_metrics(out, text)
    return mean_loss, res


def get_split_sizes(dataset_len, train_size, val_size, test_size):
    assert train_size + val_size + test_size == 1, 'The sum of the sizes must be 1'
    train_size = int(0.8 * dataset_len)
    val_size = (dataset_len - train_size) // 2
    test_size = dataset_len - train_size - val_size
    return train_size, val_size, test_size


if __name__ == '__main__':

    """
    ########### DATA ###########
    """
    tokenizer = CharacterTokenizer()
    transform = nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    dataset = FoodDataset(
        data_path=consts.DATA_PATH,
        tokenizer=tokenizer,
        transform=transform
    )

    # characters = set()
    # for item in dataset:
    #     for char in item:
    #         characters.add(char)

    
    # print(sorted(list(characters)))
    # assert False
        

    
    # Split 80% train, 10% val, 10% test
    train_size, val_size, test_size = get_split_sizes(len(dataset), 0.8, 0.1, 0.1)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(consts.SEED)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    """
    ########### MODEL ###########"
    """
    model = BaselineModel(tokenizer=tokenizer)
    
    """
    ########### OPTIMIZER ###########"
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    """
    ########### TRAINING ###########"
    """
    train(
        model,
        train_loader, val_loader, test_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        epochs=10,
        # device=torch.device('cpu')
    )

    # enc = tokenizer.encode('Hello')
    # print(enc)
    # print(len(enc))
    # dec = tokenizer.decode(enc)
    # print(dec)
    # print(len(dec))


    # out = model(torch.randn(2, 3, 224, 224))
    # print(out.shape)