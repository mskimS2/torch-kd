import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class LogitsKDLoss(nn.Module):
    """
    Do Deep Nets Really Need to be Deep?
    - http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep
    """

    def __init__(self, weight: float = 1.0, **kwargs):
        """
        Initializes the LogitsKDLoss module.

        Args:
        - weight (float, optional): Weight factor to scale the MSE loss (Default is 1.0).
        """
        super(LogitsKDLoss, self).__init__()
        self.weight = weight

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
        - torch.Tensor: The computed weighted MSE loss between student and teacher logits.
        """
        return F.mse_loss(pred_s, pred_t.detach()) * self.weight
