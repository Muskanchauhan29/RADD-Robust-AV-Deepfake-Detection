"""Step 5: L_pair, Eq. (4): (1/T) sum_t | c_t(degraded) - sg( c_t(clean) ) |, REAL clips only.
Acts on the frame-level score c(t), not on embeddings. The clean score is a stop-gradient target.
A symmetric (no stop-gradient) version is only a possible extra ablation (D-12), not implemented."""
import torch


def pair_loss(c_clean: torch.Tensor, c_deg: torch.Tensor, is_real: torch.Tensor) -> torch.Tensor:
    if c_clean.shape != c_deg.shape or c_clean.ndim != 2:
        raise ValueError("c_clean and c_deg must both be [B, T]")
    gap = (c_deg - c_clean.detach()).abs().mean(1)     # mean over T, per clip
    m = is_real.float()
    return (gap * m).sum() / m.sum().clamp(min=1)      # real clips only; 0 if the batch has none
