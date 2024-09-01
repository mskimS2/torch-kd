from torch import nn
from typing import Dict, Any, Type
from kd.logit import LogitsKDLoss
from kd.soft_target import SoftTargetKDLoss
from kd.hint import HintKDLoss
from kd.attention_transfer import AttentionTransferKDLoss
from kd.similarity_preserving import SimilarityPreservingKDLoss


def get_kd_loss(kd: str, config: Dict[str, Any]) -> nn.Module:
    """
    Returns the knowledge distillation loss based on the specified `kd` method.

    Args:
    - kd (str): Knowledge distillation method to use.

    Returns:
    - nn.Module: Knowledge distillation loss module.
    """
    kd_loss: Dict[str, Type[nn.Module]] = {
        "logits": LogitsKDLoss,
        "soft_target": SoftTargetKDLoss,
        "hints": HintKDLoss,
        "attention_transfer": AttentionTransferKDLoss,
        "similarity_preserving": SimilarityPreservingKDLoss,
    }

    if kd not in kd_loss:
        raise ValueError(f"Invalid knowledge distillation method: {kd}")

    return kd_loss[kd](**config)


__all__ = [LogitsKDLoss, SoftTargetKDLoss, HintKDLoss, AttentionTransferKDLoss, SimilarityPreservingKDLoss]
