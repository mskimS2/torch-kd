import torch
from torch import nn
import torch.nn.functional as F


class LogitsKDLoss(nn.Module):
    """
    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    """

    def __init__(self, weights: float = 1.0):
        """
        Initializes the LogitsKDLoss module.

        Args:
        - weights (float, optional): Weight factor to scale the MSE loss (Default is 1.0).
        """
        super(LogitsKDLoss, self).__init__()
        self.weights = weights

    def forward(self, student_preds: torch.Tensor, teacher_preds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the weighted knowledge distillation loss.

        Args:
        - student_preds (torch.Tensor): Logits output by the student model.
        - teacher_preds (torch.Tensor): Logits output by the teacher model.

        Returns:
        - torch.Tensor: The computed weighted MSE loss between student and teacher logits.
        """
        return F.mse_loss(student_preds, teacher_preds) * self.weights
