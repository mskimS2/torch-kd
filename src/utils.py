import torch
import random
import numpy as np


def set_randomness(random_seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
    - random_seed (int): Seed value.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_model_size(model: torch.nn.Module) -> int:
    """
    Computes the total number of parameters in a model.

    Args:
    - model (torch.nn.Module): PyTorch model.

    Returns:
    - int: Total number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = total_params * 4  # byte unit of float32
    print(f"Model size: {model_size / (1024 ** 2):.2f} MB")
    return total_params
