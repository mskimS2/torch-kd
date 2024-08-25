import os
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loggers.tensorboard import TensorBoardLogger
from collections import defaultdict
from models import ResNet
from utils import set_randomness, compute_model_size
from datasets.dataset import get_dataloaders
from kd import get_kd_loss, LogitsKDLoss, SoftTargetKDLoss, HintKDLoss
from config.parse import load_yaml


def compute_metrics(
    student_preds: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> Dict[str, float]:
    loss = criterion(student_preds, labels).item()
    preds = student_preds.argmax(dim=1).cpu().numpy()
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
    teacher: nn.Module = None,
    student: nn.Module = None,
    train_loader: DataLoader = None,
    criterion: nn.Module = None,
    kd_loss: Union[LogitsKDLoss, SoftTargetKDLoss, HintKDLoss] = None,
    optimizer: torch.optim = None,
) -> Dict[str, float]:
    teacher.eval()
    student.train()

    avg_metrics = defaultdict(float)
    num_samples = len(train_loader)
    pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
    for i, (inputs, labels) in enumerate(pbar, 0):
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

        with torch.no_grad():
            pred_t, fm_t = teacher(inputs)

        pred_s, fm_s = student(inputs)

        classification_loss = criterion(pred_s, labels)
        kd = kd_loss(pred_s, pred_t, fm_s, fm_t)

        loss = classification_loss + kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics = compute_metrics(pred_s, labels, criterion)
        train_metrics["kd"] = kd
        pbar.set_postfix(train_metrics)

        for k, v in train_metrics.items():
            avg_metrics[k] += v

    return {key: value / num_samples for key, value in avg_metrics.items()}


@torch.inference_mode()
def evaluate(
    config: Dict[str, Any] = None,
    student: nn.Module = None,
    test_loader: DataLoader = None,
    criterion: nn.Module = None,
) -> Dict[str, float]:
    avg_metrics = defaultdict(float)
    num_samples = len(test_loader)
    for images, labels in test_loader:
        images, labels = images.to(config["device"]), labels.to(config["device"])
        pred_s, fm_s = student(images)

        test_metrics = compute_metrics(pred_s, labels, criterion)
        for k, v in test_metrics.items():
            avg_metrics[k] += v

    return {key: value / num_samples for key, value in avg_metrics.items()}


def train_and_evaluate(
    config: Dict[str, Any] = None,
    teacher: nn.Module = None,
    student: nn.Module = None,
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    criterion: nn.Module = None,
    kd_loss: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    logger: Optional[TensorBoardLogger] = None,
) -> None:
    best_loss = np.inf
    for epoch in range(config["num_epochs"]):
        train_metrics = train(config, teacher, student, train_loader, criterion, kd_loss, optimizer)
        print(f"Train[{epoch + 1}] " + ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
        logger.log_metrics(train_metrics, epoch, "train")

        test_metrics = evaluate(config, student, test_loader, criterion)
        print(f"Test[{epoch + 1}] " + ", ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))
        logger.log_metrics(test_metrics, epoch, "test")

        if best_loss > test_metrics["loss"]:
            best_loss = test_metrics["loss"]
            save_model(student, os.path.join(config["model_dir"], config["model_path"] + ".pth"))


def save_model(model: nn.Module, filename: str):
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train a student model using knowledge distillation.")
    args.add_argument("--kd", type=str, default="hints", help="logits, soft_target, hints")
    args.add_argument("--model", type=str, default="resnet", help="cnn, resnet")
    p = args.parse_args()

    yaml_config = load_yaml(f"src/config/{p.model}/{p.kd}.yaml")
    config = yaml_config["training"]
    teacher_config = yaml_config["teacher"]
    student_config = yaml_config["student"]
    print(config, teacher_config, student_config)

    set_randomness(config["random_seed"])  # Set random seed for reproducibility
    os.makedirs(config["model_dir"], exist_ok=True)  # Create a directory to save the model

    # Load the dataset
    dataLoader = get_dataloaders(config)
    train_loader = dataLoader["train"]
    test_loader = dataLoader["test"]

    # Create a model
    teacher = ResNet(
        in_channels=teacher_config["in_channels"],
        layers=teacher_config["layers"],
        num_classes=10,
        use_featuremap=True,
    ).to(config["device"])
    teacher.load_state_dict(torch.load(teacher_config["save_dir"]))

    student = ResNet(
        in_channels=student_config["in_channels"],
        layers=student_config["layers"],
        num_classes=10,
        use_featuremap=True,
    ).to(config["device"])

    print("teacher model size", compute_model_size(teacher))
    print("student model size", compute_model_size(student))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr"])
    kd_loss = get_kd_loss(p.kd, config)

    # Create a logger
    logger = TensorBoardLogger(config["log_dir"])
    logger.init_logger()
    logger.init_experiment(config["experiment_name"])
    logger.log_params(config)

    # Train the model
    train_and_evaluate(config, teacher, student, train_loader, test_loader, criterion, kd_loss, optimizer, logger)
