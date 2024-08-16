import torch
from torch import nn
import torch.nn.functional as F


class SoftTargetKDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    - https://arxiv.org/pdf/1503.02531
    """

    def __init__(self, T: float = 1.0, weights: float = 1.0):
        """
        Initializes the SoftTargetKDLoss module.

        Args:
        - T (float, optional): Temperature factor to soften the student logits (Default is 1.0).
        - weights (float, optional): Weight factor to scale the MSE loss (Default is 1.0).
        """
        super(SoftTargetKDLoss, self).__init__()
        self.T = T
        self.weights = weights

    def forward(self, student_preds: torch.Tensor, teacher_preds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the weighted knowledge distillation loss.

        Args:
        - student_preds (torch.Tensor): Logits output by the student model.
        - teacher_preds (torch.Tensor): Logits output by the teacher model.

        Returns:
        - torch.Tensor: The computed weighted `soft_target` loss between student and teacher logits.
        """
        return (
            F.kl_div(
                F.log_softmax(student_preds / self.T, dim=1),
                F.softmax(teacher_preds / self.T, dim=1),
                reduction="batchmean",
            )
            * self.T**2
        ) * self.weights
