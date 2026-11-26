"""Align audio features to the video frame rate by adaptive average pooling."""
import torch
import torch.nn.functional as F


def pool_to_T(h: torch.Tensor, T: int = 50) -> torch.Tensor:
    """[N, D] -> [T, D]. For Wav2Vec 2.0 N is about 99 (50 Hz, 2 s) and T = 50 (25 fps).
    Layer choice and the resampling rule are open (D-15); average pooling is the Guide's sketch."""
    if h.ndim != 2 or h.shape[0] < T:
        raise ValueError(f"need [N>=T, D], got {tuple(h.shape)}")
    return F.adaptive_avg_pool1d(h.T[None], T)[0].T
