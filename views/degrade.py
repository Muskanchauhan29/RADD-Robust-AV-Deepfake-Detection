"""Step 2a: degradation families that need no FFmpeg (numpy / scipy / OpenCV only).

Codec families (MP3, AAC, H.264, H.265, re-encoding chains) need FFmpeg and come in step 2b.
Ranges marked * come from the paper; the others are illustrative [Proposed] and need approval (D-14).
All functions return a NEW array and never modify their input.
"""
import cv2
import numpy as np
from scipy.signal import fftconvolve

TRAIN_FAMILIES = {
    "audio": {"noise": {"snr_db": (5, 25)},          # * SNR 25-5 dB
              "mp3": {"bitrate_k": (32, 128)}},      # illustrative (step 2b)
    "video": {"h264": {"crf": (23, 40)},             # * CRF 23-40 (step 2b)
              "gblur": {"sigma": (0.5, 2.0)}},       # illustrative
}
HELDOUT_FAMILIES = {                                  # validation / test only
    "audio": ["reverb", "aac"],
    "video": ["h265", "motion_blur"],
    "joint": ["re_encoding_chain"],                   # exact chain: D-14
}


# ---------------------------------------------------------------- audio
def add_noise_snr(wav: np.ndarray, snr_db: float, rng) -> np.ndarray:
    noise = rng.standard_normal(wav.shape)
    p_sig = np.mean(wav ** 2) + 1e-12
    scale = np.sqrt(p_sig / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return wav + scale * noise


def synthetic_rir(sr: int, rt60: float, rng) -> np.ndarray:
    """Exponentially decaying noise. FOR TESTING ONLY: the real RIR source is undecided (D-14)."""
    n = int(sr * rt60)
    t = np.arange(n) / sr
    rir = rng.standard_normal(n) * 10 ** (-3 * t / rt60)    # -60 dB at t = rt60
    rir[0] = 1.0                                             # direct path
    return rir / np.sqrt(np.sum(rir ** 2))


def reverb(wav: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a room impulse response; keep length and the input RMS level."""
    out = fftconvolve(wav, rir, mode="full")[: len(wav)]
    return out * (np.sqrt(np.mean(wav ** 2) + 1e-12) / np.sqrt(np.mean(out ** 2) + 1e-12))


# ---------------------------------------------------------------- video
# frames: uint8 array [T, H, W, C]
def gaussian_blur(frames: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in frames]).reshape(frames.shape)


def motion_blur(frames: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    """Linear motion blur: averaging kernel of `length` pixels along `angle_deg`."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    return np.stack([cv2.filter2D(f, -1, k) for f in frames]).reshape(frames.shape)


# ---------------------------------------------------------------- bookkeeping
def sample_train_params(stream: str, family: str, rng) -> dict:
    """Draw one parameter set uniformly from the training range of a training family."""
    spec = TRAIN_FAMILIES[stream][family]
    out = {}
    for name, (lo, hi) in spec.items():
        v = rng.uniform(lo, hi)
        out[name] = int(round(v)) if name in ("crf", "bitrate_k") else float(v)
    return out


def assert_no_heldout(train_manifest) -> None:
    """The training manifest must contain no held-out degradation (checked by test_views.py)."""
    held = set(sum(HELDOUT_FAMILIES.values(), []))
    bad = set(train_manifest["degradation"]) & held
    assert not bad, f"held-out degradation found in training manifest: {bad}"