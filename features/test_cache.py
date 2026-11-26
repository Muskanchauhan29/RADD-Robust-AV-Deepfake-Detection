import numpy as np
import pytest
import torch
from features import cache, standins
from features.extract import extract_clip, extract_and_save
from features.pool import pool_to_T


def clip(event_s=1.0, seed=0):
    """Dark, quiet clip with one flash frame and one click at the same moment."""
    rng = np.random.default_rng(seed)
    frames = np.full((50, 32, 32, 3), 20, np.uint8)
    frames[int(event_s * 25)] = 250                                  # flash
    wav = 0.001 * rng.standard_normal(32000)
    c = int(event_s * 16000)
    wav[c:c + 400] += 0.8 * rng.standard_normal(400)                 # click
    return frames, wav


def test_pool_shape_and_value_range():
    h = torch.rand(99, 8)
    p = pool_to_T(h)
    assert p.shape == (50, 8) and p.min() >= h.min() and p.max() <= h.max()
    with pytest.raises(ValueError):
        pool_to_T(torch.rand(10, 8))


def test_extract_gives_50_and_50(tmp_path):
    f, w = clip()
    v, a = extract_clip(f, w, standins.video_encoder, standins.audio_encoder)
    assert v.shape[0] == 50 and a.shape[0] == 50
    assert standins.audio_encoder(w).shape[0] == 99                  # Wav2Vec-like frame count


@pytest.mark.parametrize("event_s", [0.4, 1.0, 1.6])
def test_audio_and_video_are_aligned(event_s):
    f, w = clip(event_s)
    v, a = extract_clip(f, w, standins.video_encoder, standins.audio_encoder)
    assert abs(int(v.sum(1).argmax()) - int(a.sum(1).argmax())) <= 1  # same event within one frame


def test_cache_roundtrip_half_precision_and_validation(tmp_path):
    f, w = clip()
    p = cache.cache_path(tmp_path, "train", "clean", "clip001")
    extract_and_save(p, f, w, standins.video_encoder, standins.audio_encoder)
    cache.validate_file(p)
    v, a = cache.load_features(p)
    assert v.dtype == torch.float16 and a.dtype == torch.float16 and v.shape[0] == a.shape[0] == 50


def test_bad_inputs_are_rejected(tmp_path):
    f, w = clip()
    with pytest.raises(ValueError):
        extract_clip(f[:49], w, standins.video_encoder, standins.audio_encoder)
    with pytest.raises(ValueError):
        extract_clip(f, w[:-1], standins.video_encoder, standins.audio_encoder)
    with pytest.raises(ValueError):
        cache.cache_path(tmp_path, "train", "weird", "x")
    with pytest.raises(ValueError):
        cache.save_features(tmp_path / "x.pt", torch.zeros(50, 4), torch.full((50, 4), float("nan")))
