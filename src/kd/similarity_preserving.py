import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class SimilarityPreservingKDLoss(nn.Module):
    """
    Similarity-Preserving Knowledge Distillation
    - https://arxiv.org/abs/1907.09682
    """

    def __init__(self, weight: float = 1.0, **kwargs):
        """
        Initializes the SimilarityPreservingKDLoss module.

        Args:
        - weight (float, optional): Weight factor to scale the loss (Default is 1.0).
        """
        super(SimilarityPreservingKDLoss, self).__init__()
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
        - torch.Tensor: The computed weighted `Similarity-Preserving` KD loss between student and teacher `feature_map`.
        """

        assert len(fm_s) == len(fm_t), ValueError(
            "The number of feature maps from student and teacher must be the same"
        )

        loss = 0.0
        for s, t in zip(fm_s, fm_t):
            G_S = self.compute_similarity_matrix(s)
            G_T = self.compute_similarity_matrix(t)
            loss += F.mse_loss(G_S, G_T.detach())

        return loss * self.weight

    def compute_similarity_matrix(self, f: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity-preserving transformation by normalizing the mean squared values
        of the feature map across the channel dimension.

        Args:
        - f (torch.Tensor): The input feature map.

        Returns:
        - torch.Tensor: The computed similarity-preserving transformation.
        """
        f = f.view(f.size(0), -1)
        return F.normalize(f @ f.t())
