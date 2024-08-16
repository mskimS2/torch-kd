import os
import torch
from .base import Logger
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any


class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir

    def init_logger(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            self.writer.add_text(f"params/{key}", str(value))

    def init_experiment(self, experiment_name: str):
        self.writer.add_text("experiment_name", experiment_name)

    def log_artifact(self, artifact_path: str):
        self.writer.add_text("artifact_path", artifact_path)

    def log_metrics(self, metrics: Dict[str, float], step: int, stage: str):
        for key, value in metrics.items():
            self.writer.add_scalar(f"{stage}/{key}", value, step)

    def set_tags(self, tags: Dict[str, str]):
        for key, value in tags.items():
            self.writer.add_text(f"tags/{key}", value)
