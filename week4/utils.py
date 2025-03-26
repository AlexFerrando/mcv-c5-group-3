from typing import Tuple, Dict, Any
import torch

def get_split_sizes(dataset_len: int, train_size: float, val_size: float, test_size: float) -> Tuple[int, int, int]:
    assert train_size + val_size + test_size == 1, 'The sum of the sizes must be 1'
    train_size = int(train_size * dataset_len)
    val_size = (dataset_len - train_size) // 2
    test_size = dataset_len - train_size - val_size
    return train_size, val_size, test_size


def pretty_print(metrics: Dict[str, Any], stage: str):
    """
    Pretty print evaluation metrics.

    Args:
        metrics (dict): Dictionary containing metric names and their values.
        stage (str): The stage name (e.g., "Test", "Validation").
    """
    print(f"\n{'='*20} {stage} Metrics {'='*20}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("="*(50 + len(stage)))


def freeze(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False