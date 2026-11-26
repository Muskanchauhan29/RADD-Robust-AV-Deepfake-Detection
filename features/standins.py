"""Stand-in encoders FOR TESTING THE PIPELINE ONLY. They are not Wav2Vec 2.0 and not a lip encoder.

They mimic the interface and the frame-rate behaviour of the real encoders so that cache format,
pooling and alignment can be tested without model downloads (the real lip encoder is open: D-02).
  video: [T, H, W, C] uint8 -> [T, 16]   mean grey level in a 4x4 grid per frame
  audio: [samples] float    -> [N, 16]   band energies, 400-sample window, hop 320 (as Wav2Vec 2.0, 50 Hz)
"""
import numpy as np
import torch


def video_encoder(frames: np.ndarray) -> torch.Tensor:
    g = frames.astype(np.float32).mean(axis=-1)                    # [T, H, W]
    T, H, W = g.shape
    g = g[:, : H // 4 * 4, : W // 4 * 4].reshape(T, 4, H // 4, 4, W // 4)
    return torch.from_numpy(g.mean(axis=(2, 4)).reshape(T, 16) / 255.0)


def audio_encoder(wav: np.ndarray) -> torch.Tensor:
    n = (len(wav) - 400) // 320 + 1                                 # 99 frames for 2 s
    idx = np.arange(400)[None] + 320 * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(wav[idx] * np.hanning(400), axis=1))  # [n, 201]
    bands = np.array_split(spec[:, :200], 16, axis=1)
    return torch.from_numpy(np.stack([b.mean(1) for b in bands], 1).astype(np.float32))
