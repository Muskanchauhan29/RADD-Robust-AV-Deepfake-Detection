"""Step 2b: desynchronised views of REAL clips.

Role: contrastive negatives in L_NCE ONLY. No class label, no L_cls, no L_pair.
Defaults and open choices are [Proposed] and listed under D-13 for supervisor approval.

Where each type acts (Guide Table 6 / Section 7):
  global_shift   on cached audio FEATURES at load time   [frames, D], 40 ms per frame at 25 fps
  local_warp     on the audio WAVEFORM, pre-encoded      1-D float array, 16 kHz
  speaker_swap   on the audio WAVEFORM, pre-encoded
  segment_replace on the audio WAVEFORM, pre-encoded     (D-13: Guide's earlier sketch worked on
                                                           features; Table 6 says pre-encoded, so the
                                                           waveform version is used here)
Video-side span replacement is not implemented (D-13).
"""
import numpy as np
from scipy.signal import resample_poly

FRAME_MS = 40          # 25 fps
SR = 16000


# ------------------------------------------------------------ feature level
def global_shift(a_feat: np.ndarray, max_ms: int, rng, pad: str = "roll"):
    """Shift audio features by k frames, k != 0 and |k| * 40 ms <= max_ms.

    pad = "roll" wraps around (as in the Guide sketch); "edge" repeats the border frame.
    Which one is used is open (D-13). Returns (shifted, k).
    """
    kmax = max_ms // FRAME_MS
    if kmax < 1:
        raise ValueError(f"max_ms={max_ms} is below one frame ({FRAME_MS} ms): no non-zero shift fits")
    k = int(rng.choice([i for i in range(-kmax, kmax + 1) if i != 0]))
    if pad == "roll":
        return np.roll(a_feat, k, axis=0), k
    if pad == "edge":
        out = np.empty_like(a_feat)
        if k > 0:
            out[k:], out[:k] = a_feat[:-k], a_feat[0]
        else:
            out[:k], out[k:] = a_feat[-k:], a_feat[-1]
        return out, k
    raise ValueError("pad must be 'roll' or 'edge'")


# ------------------------------------------------------------ waveform level
def local_warp(wav: np.ndarray, rng, n_segments: int = 3,
               factor_range=(0.8, 1.25)) -> np.ndarray:
    """Time-stretch / compress consecutive segments of a waveform; total length is kept.

    Stretching by resampling also changes pitch slightly (no pitch-preserving stretch is used).
    n_segments and factor_range are [Proposed] (D-13).
    """
    n = len(wav)
    bounds = np.linspace(0, n, n_segments + 1).astype(int)
    parts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        f = rng.uniform(*factor_range)
        # up/down are small integers approximating the factor
        up, down = int(round(f * 100)), 100
        parts.append(resample_poly(wav[a:b], up, down))
    out = np.concatenate(parts)
    if len(out) >= n:                          # trim or zero-pad back to the original length
        out = out[:n]
    else:
        out = np.pad(out, (0, n - len(out)))
    return out


def speaker_swap(wav: np.ndarray, donor_wav: np.ndarray) -> np.ndarray:
    """Replace the audio by that of another real clip (the caller picks a different identity)."""
    n = len(wav)
    if len(donor_wav) < n:
        raise ValueError("donor clip is shorter than the target clip")
    return donor_wav[:n].copy()


def segment_replace(wav: np.ndarray, donor_wav: np.ndarray, start: int, length: int) -> np.ndarray:
    """Replace samples [start, start + length) by the same span of a donor clip."""
    if start < 0 or start + length > len(wav) or len(donor_wav) < start + length:
        raise ValueError("span outside the clip or donor too short")
    out = wav.copy()
    out[start:start + length] = donor_wav[start:start + length]
    return out


# ------------------------------------------------------------ bookkeeping
def assert_real_only(manifest) -> None:
    """Desynchronised rows must belong to real clips (label 0). Needs columns label, view."""
    bad = manifest[(manifest["view"] == "desync") & (manifest["label"] != 0)]
    assert bad.empty, f"desynchronised views of fake clips found: {len(bad)} rows"