import numpy as np
import pandas as pd
import pytest
from views import desync as S

SR = S.SR


def rng(s=0):
    return np.random.default_rng(s)


def feats():
    return np.arange(50 * 4, dtype=np.float32).reshape(50, 4)       # each row unique


@pytest.mark.parametrize("pad", ["roll", "edge"])
def test_global_shift_nonzero_bounded_and_shape(pad):
    a = feats()
    for s in range(30):
        out, k = S.global_shift(a, 200, rng(s), pad)
        assert k != 0 and abs(k) * S.FRAME_MS <= 200
        assert out.shape == a.shape and not np.array_equal(out, a)


def test_global_shift_is_a_true_shift():
    a = feats()
    out, k = S.global_shift(a, 200, rng(1), "roll")
    assert np.array_equal(out, np.roll(a, k, axis=0))
    out, k = S.global_shift(a, 200, rng(1), "edge")
    if k > 0:
        assert np.array_equal(out[k:], a[:-k]) and np.all(out[:k] == a[0])
    else:
        assert np.array_equal(out[:k], a[-k:]) and np.all(out[k:] == a[-1])


def test_global_shift_rejects_margin_below_one_frame():
    with pytest.raises(ValueError):
        S.global_shift(feats(), 30, rng())


def test_local_warp_keeps_length_changes_signal_and_input():
    wav = rng(2).standard_normal(2 * SR); before = wav.copy()
    out = S.local_warp(wav, rng(3))
    assert out.shape == wav.shape and not np.allclose(out, wav)
    assert np.array_equal(wav, before) and np.isfinite(out).all()


def test_speaker_swap_returns_donor_audio():
    wav, donor = rng(4).standard_normal(2 * SR), rng(5).standard_normal(3 * SR)
    out = S.speaker_swap(wav, donor)
    assert np.array_equal(out, donor[:2 * SR]) and out is not donor
    with pytest.raises(ValueError):
        S.speaker_swap(donor, wav)


def test_segment_replace_only_changes_the_span():
    wav, donor = rng(6).standard_normal(2 * SR), rng(7).standard_normal(2 * SR)
    out = S.segment_replace(wav, donor, 8000, 4000)
    assert np.array_equal(out[:8000], wav[:8000]) and np.array_equal(out[12000:], wav[12000:])
    assert np.array_equal(out[8000:12000], donor[8000:12000])
    with pytest.raises(ValueError):
        S.segment_replace(wav, donor, 2 * SR - 10, 100)


def test_desync_rows_must_be_real():
    ok = pd.DataFrame({"view": ["clean", "desync"], "label": [1, 0]})
    S.assert_real_only(ok)
    bad = pd.DataFrame({"view": ["desync"], "label": [1]})
    with pytest.raises(AssertionError):
        S.assert_real_only(bad)