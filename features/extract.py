"""Turn one clip (frames + 16 kHz waveform) into aligned cached features."""
import numpy as np
import torch
from features.cache import T, save_features
from features.pool import pool_to_T


def extract_clip(frames: np.ndarray, wav: np.ndarray, video_encoder, audio_encoder):
    """Returns (video [T, Dv], audio [T, Da]); raises if the clip is not 2 s at 25 fps / 16 kHz."""
    if frames.shape[0] != T:
        raise ValueError(f"expected {T} frames, got {frames.shape[0]}")
    if len(wav) != 2 * 16000:
        raise ValueError(f"expected 32000 samples, got {len(wav)}")
    v = video_encoder(frames)
    a = pool_to_T(audio_encoder(wav), T)
    return v, a


def extract_and_save(path, frames, wav, video_encoder, audio_encoder) -> None:
    v, a = extract_clip(frames, wav, video_encoder, audio_encoder)
    save_features(path, v, a)
