import numpy as np
import pandas as pd
import pytest
import build_features as B
from features import standins
from features.cache import cache_path, load_features, T

EVALS = ["clean", "degraded", "heldout_audio", "heldout_video", "heldout_joint"]


@pytest.fixture
def clips(tmp_path):
    rng = np.random.default_rng(0); rows = []
    for i, (split, y) in enumerate([("train", 0), ("train", 1), ("val", 0), ("test", 0), ("test", 1)]):
        p = tmp_path / f"c{i}.npz"
        np.savez(p, frames=rng.integers(0, 255, (50, 32, 32, 3), dtype=np.uint8), wav=(0.1 * rng.standard_normal(32000)).astype(np.float32))
        rows.append(dict(clip_id=f"v{i}_c0", video_id=f"v{i}", identity_id=f"p{i % 3}", y=y, category="x", split=split, path=str(p)))
    return pd.DataFrame(rows)


def test_views_per_split_and_shapes(clips, tmp_path):
    n = B.build(clips, tmp_path / "cache", standins.video_encoder, standins.audio_encoder)
    assert n == 3 * 2 + 2 * 5                           # train/val: 2 views, test: 5 views
    for r in clips.itertuples():
        for v in (EVALS if r.split == "test" else EVALS[:2]):
            video, audio = load_features(cache_path(tmp_path / "cache", r.split, v, r.clip_id))
            assert video.shape[0] == T and audio.shape[0] == T
        if r.split != "test":
            assert not cache_path(tmp_path / "cache", r.split, "heldout_audio", r.clip_id).exists()   # held-out only in test


def test_degraded_differs_from_clean_and_is_reproducible(clips):
    r = clips.iloc[0]; d = np.load(r.path)
    a = B.make_views(r.clip_id, d["frames"], d["wav"], "train"); b = B.make_views(r.clip_id, d["frames"], d["wav"], "train")
    assert not np.allclose(a["clean"][1], a["degraded"][1]) and not np.array_equal(a["clean"][0], a["degraded"][0])
    assert np.array_equal(a["degraded"][1], b["degraded"][1])
    assert set(a) == {"clean", "degraded"}


def test_inputs_not_modified_and_heldout_effects(clips):
    r = clips.iloc[3]; d = np.load(r.path); f0, w0 = d["frames"].copy(), d["wav"].copy()
    v = B.make_views(r.clip_id, d["frames"], d["wav"], "test")
    assert np.array_equal(d["frames"], f0) and np.array_equal(d["wav"], w0)
    assert np.array_equal(v["heldout_audio"][0], f0) and np.array_equal(v["heldout_video"][1], w0)      # one stream each
    assert not np.array_equal(v["heldout_joint"][0], f0) and not np.allclose(v["heldout_joint"][1], w0)


def test_manifest_has_no_heldout_in_training(clips, tmp_path):
    from views.degrade import assert_no_heldout
    m = B.write_manifest(clips, tmp_path / "manifests" / "clips.csv")
    assert_no_heldout(m[m.split == "train"]) is None
    assert list(pd.read_csv(tmp_path / "manifests" / "clips.csv").columns) == \
        ["clip_id", "video_id", "identity_id", "y", "category", "split", "degradation"]


def test_real_encoders_module_imports_without_heavy_deps():
    import features.real_encoders as R            # heavy imports happen only when the loaders are called
    assert callable(R.load_audio_encoder) and callable(R.load_video_encoder)