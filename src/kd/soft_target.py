import torch
from torch import nn
import torch.nn.functional as F
from typing import List


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

    def forward(
        self,
        pred_s: torch.Tensor = None,
        pred_t: torch.Tensor = None,
        fm_s: List[torch.Tensor] = None,
        fm_t: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute the weighted knowledge distillation loss.

        Args:
        - pred_s (torch.Tensor): Logits output by the student model.
        - pred_t (torch.Tensor): Logits output by the teacher model.
        - fm_s (List[torch.Tensor], optional): Feature maps output by the student model (Default is None).
        - fm_t (List[torch.Tensor], optional): Feature maps output by the teacher model (Default is None).

        Returns:
        - torch.Tensor: The computed weighted `soft_target` loss between student and teacher logits.
        """
        return (
            F.kl_div(
                F.log_softmax(pred_s / self.T, dim=1),
                F.softmax(pred_t.detach() / self.T, dim=1),
                reduction="batchmean",
            )
            * self.T**2
        ) * self.weights
