import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loggers.tensorboard import TensorBoardLogger
from collections import defaultdict
from models import CNN, SmallCNN, ResNet
from utils import set_randomness, compute_model_size
from datasets.dataset import get_dataloaders
from config.parse import load_yaml


def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor, loss: float) -> Dict[str, float]:
    loss = criterion(outputs, labels).item()
    preds = outputs.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def train(
    config: Dict[str, Any] = None,
    model: nn.Module = None,
    train_loader: DataLoader = None,
    criterion: nn.Module = None,
    optimizer: torch.optim = None,
) -> Dict[str, float]:
    model.train()

    avg_metrics = defaultdict(float)
    num_samples = len(train_loader)
    pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
    for i, (inputs, labels) in enumerate(pbar, 0):
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics = compute_metrics(outputs, labels, criterion)
        pbar.set_postfix(train_metrics)

        for k, v in train_metrics.items():
            avg_metrics[k] += v

    return {key: value / num_samples for key, value in avg_metrics.items()}


@torch.inference_mode()
def evaluate(
    config: Dict[str, Any] = None,
    model: nn.Module = None,
    test_loader: DataLoader = None,
    criterion: nn.Module = None,
) -> Dict[str, float]:
    avg_metrics = defaultdict(float)
    num_samples = len(test_loader)
    for images, labels in test_loader:
        images, labels = images.to(config["device"]), labels.to(config["device"])
        outputs, _ = model(images)

        test_metrics = compute_metrics(outputs, labels, criterion)
        for k, v in test_metrics.items():
            avg_metrics[k] += v

    return {key: value / num_samples for key, value in avg_metrics.items()}


def train_and_evaluate(
    config: Dict[str, Any] = None,
    model: nn.Module = None,
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    criterion: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    logger: Optional[TensorBoardLogger] = None,
    scheduler: torch.optim.Optimizer = None,
) -> None:
    best_loss = np.inf
    for epoch in range(config["num_epochs"]):
        train_metrics = train(config, model, train_loader, criterion, optimizer)
        print(f"Train[{epoch + 1}] " + ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
        logger.log_metrics(train_metrics, epoch, "train")

        if scheduler is not None:
            scheduler.step()

        test_metrics = evaluate(config, model, test_loader, criterion)
        print(f"Test[{epoch + 1}] " + ", ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))
        logger.log_metrics(test_metrics, epoch, "test")

        if best_loss > test_metrics["loss"]:
            best_loss = test_metrics["loss"]
            save_model(model, os.path.join(config["model_dir"], config["model_path"] + ".pth"))


def save_model(model: nn.Module, filename: str):
    torch.save(model.state_dict(), filename)


def load_model(filename: str, num_classes: int) -> nn.Module:
    model = CNN(num_classes)
    model.load_state_dict(torch.load(filename))
    return model


if __name__ == "__main__":
    config = load_yaml("src/config/resnet/student.yaml")
    # config = load_yaml("src/config/resnet/teacher.yaml")
    print(config)

    # Set random seed for reproducibility
    set_randomness(config["random_seed"])

    # Load the dataset
    dataLoader = get_dataloaders(config)
    train_loader = dataLoader["train"]
    test_loader = dataLoader["test"]

    # Create a model
    model = ResNet(
        in_channels=config["in_channels"],
        layers=config["layers"],
        num_classes=10,
        use_featuremap=False,
    ).to(config["device"])

    print(compute_model_size(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Create a directory to save the model
    os.makedirs(config["model_dir"], exist_ok=True)

    # Create a logger
    logger = TensorBoardLogger(config["log_dir"])
    logger.init_logger()
    logger.init_experiment(config["experiment_name"])
    logger.log_params(config)

    # Train the model
    train_and_evaluate(config, model, train_loader, test_loader, criterion, optimizer, logger, scheduler)
