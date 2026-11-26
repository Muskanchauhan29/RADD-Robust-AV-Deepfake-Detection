import numpy as np
import pandas as pd
import pytest
from views import degrade as D


def rng(s=0):
    return np.random.default_rng(s)


def test_noise_hits_target_snr():
    wav = np.sin(2 * np.pi * 220 * np.arange(32000) / 16000)
    for snr in (5, 15, 25):
        noisy = D.add_noise_snr(wav, snr, rng())
        measured = 10 * np.log10(np.mean(wav ** 2) / np.mean((noisy - wav) ** 2))
        assert abs(measured - snr) < 0.2


def test_reverb_keeps_length_level_and_changes_signal():
    wav = rng(1).standard_normal(32000)
    out = D.reverb(wav, D.synthetic_rir(16000, 0.4, rng(2)))
    assert out.shape == wav.shape
    assert abs(np.sqrt(np.mean(out ** 2)) - np.sqrt(np.mean(wav ** 2))) < 1e-6
    assert not np.allclose(out, wav)


def _frames():
    f = rng(3).integers(0, 256, size=(4, 64, 64, 3), dtype=np.uint8)
    return f


def _sharp(frames):                       # variance of Laplacian: lower = blurrier
    return np.mean([cv2_lap(f) for f in frames])


def cv2_lap(f):
    import cv2
    return cv2.Laplacian(f, cv2.CV_64F).var()


@pytest.mark.parametrize("fn,args", [(D.gaussian_blur, (1.5,)), (D.motion_blur, (9, 30.0))])
def test_blur_reduces_sharpness_keeps_shape_and_input(fn, args):
    f = _frames(); before = f.copy()
    out = fn(f, *args)
    assert out.shape == f.shape and out.dtype == np.uint8
    assert _sharp(out) < 0.5 * _sharp(f)
    assert np.array_equal(f, before)       # input not modified


def test_motion_blur_is_directional():
    f = np.zeros((1, 64, 64, 1), np.uint8); f[0, :, 32, 0] = 255        # vertical line
    horiz = D.motion_blur(f, 9, 0.0)[0, :, :, 0]
    vert = D.motion_blur(f, 9, 90.0)[0, :, :, 0]
    assert horiz[32, 28:37].std() < f[0, 32, 28:37, 0].std()           # horizontal blur smears the line
    assert np.abs(vert.astype(int) - f[0, :, :, 0]).max() < 40          # vertical blur leaves it almost unchanged


def test_params_inside_training_ranges_and_deterministic():
    for stream, fams in D.TRAIN_FAMILIES.items():
        for fam, spec in fams.items():
            p = D.sample_train_params(stream, fam, rng(5))
            for k, (lo, hi) in spec.items():
                assert lo <= p[k] <= hi
            assert p == D.sample_train_params(stream, fam, rng(5))


def test_training_and_heldout_family_names_are_disjoint():
    train = {f for fams in D.TRAIN_FAMILIES.values() for f in fams}
    held = set(sum(D.HELDOUT_FAMILIES.values(), []))
    assert not train & held


def test_heldout_guard():
    ok = pd.DataFrame({"degradation": ["none", "noise", "h264", "gblur"]})
    D.assert_no_heldout(ok)
    bad = pd.DataFrame({"degradation": ["none", "reverb"]})
    with pytest.raises(AssertionError):
        D.assert_no_heldout(bad)