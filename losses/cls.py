"""Step 5: L_cls. Binary cross-entropy on the clip logit (real = 0, fake = 1).
Degradation never changes the label, so degraded copies are passed in with the same y."""
import torch
import torch.nn.functional as F


def cls_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.shape != y.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(y.shape)} must match")
    return F.binary_cross_entropy_with_logits(logits, y.float())
