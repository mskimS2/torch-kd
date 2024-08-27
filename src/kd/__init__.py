from torch import nn
from typing import Dict, Any
from kd.logit import LogitsKDLoss
from kd.soft_target import SoftTargetKDLoss
from kd.hint import HintKDLoss
from kd.attention_transfer import AttentionTransferKDLoss


def get_kd_loss(kd: str, config: Dict[str, Any]) -> nn.Module:
    """
    Returns the knowledge distillation loss based on the specified `kd` method.

    Args:
    - kd (str): Knowledge distillation method to use.

    Returns:
    - nn.Module: Knowledge distillation loss module.
    """
    if kd == "logits":
        return LogitsKDLoss(config["kd_weights"])
    elif kd == "soft_target":
        return SoftTargetKDLoss(config["temperature"], config["kd_weights"])
    elif kd == "hints":
        return HintKDLoss(config["kd_weights"])
    elif kd == "attention_transfer":
        return AttentionTransferKDLoss(config["kd_pow"], config["kd_weights"], config["kd_type"])

    raise ValueError(f"Invalid knowledge distillation method: {kd}")


__all__ = [
    LogitsKDLoss,
    SoftTargetKDLoss,
    HintKDLoss,
    AttentionTransferKDLoss,
]
