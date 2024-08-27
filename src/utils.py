import torch
import random
import numpy as np
from torch import nn
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def compute_metrics(pred_s: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> Dict[str, float]:
    """
    Computes the classification metrics.

    Args:
    - pred_s (torch.Tensor): Predictions made by the student model.
    - labels (torch.Tensor): Ground truth labels.
    - criterion (nn.Module): Loss function.

    Returns:
    - Dict[str, float]: Classification metrics.
    """
    loss = criterion(pred_s, labels).item()
    preds = pred_s.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(labels, preds, average="weighted", zero_division=0),
    }
