import subprocess
import numpy as np
import pandas as pd
import pytest
from data.make_synthetic_manifest import make
from data.split_fakeavceleb import select_subset, split_by_identity, split_counts, check_split
from data import make_clips as MC
from views.codecs import have_ffmpeg

RATIOS = {"train": .70, "val": .15, "test": .15}


def test_subset_sizes_identities_and_categories():
    sub = select_subset(make(60), 10, 10, seed=0)
    assert (sub.label == 0).sum() == 10 and (sub.label == 1).sum() == 10
    assert sub[sub.label == 0].identity_id.nunique() == 10                     # one real video per identity
    assert set(sub[sub.label == 1].identity_id) <= set(sub[sub.label == 0].identity_id)
    assert sub[sub.label == 1].category.nunique() >= 2                         # spread over categories


def test_subset_is_deterministic_and_splits_without_leak():
    a, b = select_subset(make(60), 10, 10, seed=1), select_subset(make(60), 10, 10, seed=1)
    pd.testing.assert_frame_equal(a, b)
    df = split_by_identity(a, RATIOS, seed=1)
    check_split(df, min_real=1)
    print(split_counts(df).to_string())


def test_subset_raises_when_not_enough_data():
    with pytest.raises(ValueError):
        select_subset(make(5), 10, 10, seed=0)


def test_default_guard_still_stops_tiny_pilot():
    df = split_by_identity(select_subset(make(60), 10, 10, seed=0), RATIOS, seed=0)
    with pytest.raises(AssertionError):
        check_split(df, min_real=5)               # tiny pilot is refused unless the guard is lowered on purpose


def test_windows_and_no_default_crop():
    assert MC.n_windows(160, 200000) == 3 and MC.n_windows(49, 10**6) == 0 and MC.n_windows(500, 10**6, 2) == 2
    with pytest.raises(ValueError):
        MC.make_clips(pd.DataFrame(), "x", crop=None)


def test_center_crop_shape():
    out = MC.center_crop(np.zeros((50, 80, 120, 3), np.uint8), 32)
    assert out.shape == (50, 32, 32, 3)


@pytest.fixture
def video(tmp_path):
    if not have_ffmpeg():
        pytest.skip("ffmpeg missing")
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=5.3",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=5.3", "-shortest", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_cut_real_video_into_clips(video, tmp_path):
    m = pd.DataFrame([dict(video_id="v", identity_id="i", label=0, category="real", path=str(video), split="train")])
    df = MC.make_clips(m, tmp_path / "clips", "center", size=32)
    assert len(df) == 2                                                   # 5.3 s -> two full 2 s windows
    d = np.load(df.path[0])
    assert d["frames"].shape == (50, 32, 32, 3) and d["wav"].shape == (32000,)
    assert list(df.columns) == ["clip_id", "video_id", "identity_id", "y", "category", "split", "start_s", "path"]
    assert (tmp_path / "clips" / "clips.csv").exists()
