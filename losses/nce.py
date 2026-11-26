"""Step 5: L_NCE, Eq. (3), computed on REAL clips only.
-log( sum_P exp(s/tau) / ( sum_P exp(s/tau) + sum_N exp(s/tau) ) )   with tau = 0.07
"""
import torch


def nce_loss(s_pos: torch.Tensor, s_neg: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """s_pos [N, Np]: cosine similarities of matched pairs (clean; + degraded if nce_deg_pos is on).
    s_neg [N, Nn]: similarities of desynchronised pairs. N indexes (real clip, time step)."""
    if s_pos.ndim != 2 or s_neg.ndim != 2 or s_pos.shape[0] != s_neg.shape[0]:
        raise ValueError(f"expected [N, Np] and [N, Nn], got {tuple(s_pos.shape)} and {tuple(s_neg.shape)}")
    if s_pos.shape[0] == 0:
        return s_pos.sum() * 0.0                       # no real clips in the batch
    pos = torch.logsumexp(s_pos / tau, dim=1)
    allv = torch.logsumexp(torch.cat([s_pos, s_neg], 1) / tau, dim=1)
    return (allv - pos).mean()
