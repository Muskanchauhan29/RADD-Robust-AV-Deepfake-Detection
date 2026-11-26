import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import train as TR
from config import load_config
from features import cache

T, DV, DA = 50, 12, 10


def make_world(root, signal=1.5, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for split, nr, nf in (("train", 10, 30), ("val", 6, 12)):
        for i in range(nr + nf):
            y = 0 if i < nr else 1
            rows.append(dict(clip_id=f"{split}{i}", video_id=f"v{split}{i}", identity_id=f"i{split}{i}", y=y,
                             category="real" if y == 0 else "fakeA", split=split, degradation="noise"))
    df = pd.DataFrame(rows)
    for r in df.itertuples():
        for view in ("clean", "degraded"):
            v = rng.standard_normal((T, DV)) + signal * r.y
            cache.save_features(cache.cache_path(root, r.split, view, r.clip_id),
                                torch.from_numpy(v), torch.from_numpy(rng.standard_normal((T, DA))))
    return df


def small_cfg(variant, cache_dir, epochs=3, patience=5):
    cfg = load_config(variant)
    cfg.data.cache_dir = str(cache_dir)
    cfg.model.d, cfg.model.layers, cfg.model.proj_dim = 32, 1, 16
    cfg.train.batch_size, cfg.train.max_epochs, cfg.train.patience, cfg.train.lr = 8, epochs, patience, 1e-3
    cfg.desync.max_shift_ms = 200
    return cfg


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    return make_world(root), root


@pytest.mark.parametrize("variant", ["C", "D1"])
def test_run_writes_the_per_run_files_and_a_valid_history(world, tmp_path, variant):
    df, root = world
    out = TR.run_training(small_cfg(variant, root), df, tmp_path / variant, seed=0, device="cpu")
    for name in ("config.yaml", "train.log", "training_history.csv", "best.pt"):
        assert (out / name).exists(), name
    h = pd.read_csv(out / "training_history.csv")
    assert list(h.columns) == TR.HISTORY_COLS and len(h) == 3
    assert np.isfinite(h[["L_cls", "L_NCE", "total_loss", "val_auc", "val_loss"]].to_numpy()).all()
    assert h["L_pair"].isna().all() == (variant == "C")             # pair term only when the flag is on
    saved = yaml.safe_load(open(out / "config.yaml"))
    assert saved["seed"] == 0 and "git_commit" in saved and saved["config"]["flags"]["pair_loss"] == (variant == "D1")
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert ck["dims"] == {"dv": DV, "da": DA} and ck["val_auc"] == h["val_auc"].max()


def test_plumbing_learns_an_easy_synthetic_signal(world, tmp_path):
    df, root = world
    out = TR.run_training(small_cfg("D0", root, epochs=6), df, tmp_path / "learn", seed=0, device="cpu")
    h = pd.read_csv(out / "training_history.csv")
    assert h["val_auc"].max() > 0.9 and h["L_cls"].iloc[-1] < h["L_cls"].iloc[0]


def test_same_seed_reproduces_and_different_seed_differs(world, tmp_path):
    df, root = world
    run = lambda name, s: pd.read_csv(TR.run_training(small_cfg("D1b", root, 2), df, tmp_path / name, s, "cpu") / "training_history.csv")
    a, b, c = run("a", 1), run("b", 1), run("c", 2)
    cols = ["L_cls", "L_NCE", "L_pair", "val_auc"]
    assert np.allclose(a[cols], b[cols], atol=1e-6) and not np.allclose(a[cols], c[cols])


def test_refuses_to_train_when_desync_shift_is_unset(world, tmp_path):
    df, root = world
    cfg = small_cfg("C", root); cfg.desync.max_shift_ms = None
    with pytest.raises(ValueError):
        TR.run_training(cfg, df, tmp_path / "x", 0, "cpu")
    assert load_config("C").desync.max_shift_ms is None             # the shipped base config leaves D-13 unset


def test_early_stopper():
    s = TR.EarlyStopper(patience=2)
    assert s.update(0.6) == (True, False)
    assert s.update(0.5) == (False, False)
    assert s.update(0.55) == (False, True)
    assert s.update(0.9) == (True, False)


def test_early_stopping_stops_the_run(world, tmp_path):
    df, root = world
    cfg = small_cfg("C", root, epochs=30, patience=1); cfg.train.lr = 0.0   # no learning: val_auc cannot improve
    out = TR.run_training(cfg, df, tmp_path / "es", 0, "cpu")
    assert len(pd.read_csv(out / "training_history.csv")) == 2              # epoch 1 best, epoch 2 not -> stop
