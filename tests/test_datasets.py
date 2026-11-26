import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
import datasets as DS
from features import cache
from config import load_config
from model.heads import RACL
import train_step as TS

T, DV, DA = 50, 16, 24


@pytest.fixture()
def world(tmp_path):
    """Tiny synthetic cache: 6 real + 14 fake clips in train, 2 + 4 in val. Random features, no real data."""
    rng = np.random.default_rng(0)
    rows = []
    for split, n_real, n_fake in (("train", 6, 14), ("val", 2, 4)):
        for i in range(n_real + n_fake):
            y = 0 if i < n_real else 1
            rows.append(dict(clip_id=f"{split}{i}", video_id=f"v{split}{i // 2}", identity_id=f"id{split}{i // 4}",
                             y=y, category="real" if y == 0 else "fakeA", split=split, degradation="noise"))
    df = pd.DataFrame(rows)
    for r in df.itertuples():
        for view in ("clean", "degraded"):
            cache.save_features(cache.cache_path(tmp_path, r.split, view, r.clip_id),
                                torch.from_numpy(rng.standard_normal((T, DV))), torch.from_numpy(rng.standard_normal((T, DA))))
    # a pre-encoded desync file for one real clip
    cache.save_features(cache.cache_path(tmp_path, "train", "desync", "train0"),
                        torch.randn(T, DV), torch.randn(T, DA))
    return df, tmp_path


def ds(world, load_degraded=True, split="train", **kw):
    df, root = world
    return DS.ClipDataset(df, root, split, max_shift_ms=200, load_degraded=load_degraded, **kw)


def test_item_shapes_and_fake_placeholder(world):
    d = ds(world)
    real, fake = d[1], d[10]
    assert real["v_clean"].shape == (T, DV) and real["a_clean"].shape == (T, DA)
    assert real["v_clean"].dtype == torch.float32 and real["y"] == 0 and fake["y"] == 1
    assert fake["v_desync"].abs().sum() == 0 and fake["a_desync"].abs().sum() == 0
    assert not torch.equal(real["a_desync"], real["a_clean"])


def test_global_shift_is_applied_at_load_time_within_the_limit(world):
    d = ds(world)
    for i in range(1, 6):                                      # real clips without a cached desync file
        it = d[i]
        a, s = it["a_clean"], it["a_desync"]
        ks = [k for k in range(-5, 6) if k and torch.equal(torch.roll(a, k, 0), s)]
        assert ks and torch.equal(it["v_desync"], it["v_clean"])   # audio shifted, video untouched


def test_cached_desync_file_is_used_sometimes(world):
    d = ds(world)
    cached = cache.load_features(cache.cache_path(world[1], "train", "desync", "train0"))[1].float()
    used = 0
    for e in range(40):
        d.set_epoch(e); used += torch.allclose(d[0]["a_desync"], cached)
    assert 5 < used < 35                                       # roughly half of the draws


def test_variant_C_style_dataset_never_opens_degraded_files(world, monkeypatch):
    opened = []
    orig = cache.load_features
    monkeypatch.setattr(DS, "load_features", lambda p: (opened.append(str(p)), orig(p))[1])
    d = ds(world, load_degraded=False)
    for i in range(len(d)):
        it = d[i]; assert "v_deg" not in it
    assert opened and not any("/degraded/" in p for p in opened)


def test_missing_max_shift_and_heldout_training_manifest_are_rejected(world):
    df, root = world
    with pytest.raises(ValueError):
        DS.ClipDataset(df, root, "train", max_shift_ms=None, load_degraded=True)
    bad = df.copy(); bad.loc[0, "degradation"] = "reverb"
    with pytest.raises(AssertionError):
        DS.ClipDataset(bad, root, "train", max_shift_ms=200, load_degraded=True)
    DS.ClipDataset(bad, root, "val", max_shift_ms=200, load_degraded=True)   # allowed outside training (D-10)


def test_dataset_is_reproducible_for_same_seed_and_epoch(world):
    a, b = ds(world, seed=3), ds(world, seed=3)
    assert torch.equal(a[2]["a_desync"], b[2]["a_desync"])
    b.set_epoch(1); c = [torch.equal(a[i]["a_desync"], b[i]["a_desync"]) for i in range(1, 6)]
    assert not all(c)


def test_sampler_is_balanced_covers_every_fake_once_and_is_deterministic(world):
    d = ds(world); labels = d.df.y.to_numpy()
    s = DS.BalancedBatchSampler(labels, batch_size=4, seed=0)
    batches = list(s)
    assert len(batches) == len(s) == 7                           # 14 fakes / 2 per batch
    for b in batches:
        assert len(b) == 4 and labels[b].sum() == 2              # exactly half fake
    fakes = [i for b in batches for i in b if labels[i] == 1]
    assert sorted(fakes) == sorted(np.flatnonzero(labels == 1)) # each fake once
    assert batches == list(DS.BalancedBatchSampler(labels, 4, seed=0))
    s.set_epoch(1); assert batches != list(s)
    with pytest.raises(ValueError):
        DS.BalancedBatchSampler(labels, 5)
    with pytest.raises(ValueError):
        DS.BalancedBatchSampler(np.ones(4), 4)


def test_real_clips_are_oversampled_when_few(world):
    d = ds(world); labels = d.df.y.to_numpy()                     # 6 real clips, 8 real slots per 16-batch
    s = DS.BalancedBatchSampler(labels, batch_size=16)
    b = next(iter(s)); assert sum(labels[i] == 0 for i in b) == 8 and len(set(b)) < 16


@pytest.mark.parametrize("variant", ["C", "D0", "D1a", "D1b", "D1"])
def test_real_dataloader_batch_runs_through_training_step(world, variant):
    cfg = load_config(variant); f = cfg.flags
    d = ds(world, load_degraded=f.degraded_cls or f.nce_deg_pos or f.pair_loss)
    dl = DataLoader(d, batch_sampler=DS.BalancedBatchSampler(d.df.y.to_numpy(), 4), collate_fn=DS.collate)
    batch = next(iter(dl))
    m = RACL(DV, DA, d=32, layers=1)
    loss, parts = TS.training_step(m, batch, cfg)
    loss.backward()
    assert torch.isfinite(loss) and batch["y"].sum() == 2
