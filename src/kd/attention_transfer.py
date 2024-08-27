import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class AttentionTransferKDLoss(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    - https://arxiv.org/abs/1612.03928
    """

    def __init__(self, p: float = 2.0, weight: float = 1.0, loss_type: "str" = "sum"):
        """
        Initializes the AttentionTransferKDLoss module.

        Args:
        - p (float, optional): Pow factor to soften the feature map values (Default is 2.0).
        - weight (float, optional): Weight factor to scale the loss (Default is 1.0).
        - loss_type (str, optional): Type of reduction to apply to the loss. Options are "sum" or "max" (Default is "sum").
        """
        super(AttentionTransferKDLoss, self).__init__()
        self.p = p
        self.weight = weight
        self.loss_type = loss_type

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
        - torch.Tensor: The computed weighted `attention_transfer` loss between student and teacher `feature_map`.
        """

        assert len(fm_s) == len(fm_t), ValueError(
            "The number of feature maps from student and teacher must be the same"
        )

        if self.loss_type == "max":
            return max([F.mse_loss(self.at(s), self.at(t).detach()) for s, t in zip(fm_s, fm_t)]) * self.weight

        return sum([F.mse_loss(self.at(s), self.at(t).detach()) for s, t in zip(fm_s, fm_t)]) * self.weight

    def at(self, f: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention map by normalizing the mean squared values
        of the feature map across the channel dimension.

        Args:
        - f (torch.Tensor): Input feature map tensor with shape (B, C, H, W).

        Returns:
        - torch.Tensor:
            Normalized attention map with shape (B, H*W).
            Square the tensor, average across channels (C), and flatten the spatial dimensions (H, W).
        """
        return F.normalize(f.pow(self.p).mean(dim=1).view(f.size(0), -1))
