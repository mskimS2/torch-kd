import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class HintKDLoss(nn.Module):
    """
    FitNets: Hints for Thin Deep Nets
    - https://arxiv.org/pdf/1412.6550
    """

    def __init__(self, weight: float = 1.0):
        """
        Initializes the HintKDLoss module.

        Args:
        - weight (float, optional): Weight factor to scale the loss (Default is 1.0).
        """
        super(HintKDLoss, self).__init__()
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
        - pred_s (List[torch.Tensor]): Logits output by the student model.
        - pred_t (List[torch.Tensor]): Logits output by the teacher model.
        - fm_s (List[torch.Tensor], optional): Feature maps output by the student model (Default is None).
        - fm_t (List[torch.Tensor], optional): Feature maps output by the teacher model (Default is None).

        Returns:
        - torch.Tensor: The computed weighted `hint loss` loss between student and teacher logits.
        """

        assert len(fm_s) == len(fm_t), ValueError(
            "The number of feature maps from student and teacher must be the same"
        )

        return F.mse_loss(fm_s[-1], fm_t[-1].detach()) * self.weight
