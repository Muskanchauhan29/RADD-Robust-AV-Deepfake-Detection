"""Step 8a: cache-backed dataset and balanced batch sampler.

Clip manifest columns: clip_id, video_id, identity_id, y (0 real / 1 fake), category, split, degradation.
Features come from features/cache/{split}/{view}/{clip_id}.pt (views: clean, degraded, desync).

Batch keys (match train_step.py): v_clean, a_clean, [v_deg, a_deg], v_desync, a_desync, y.
- Degraded keys are present only if load_degraded is True (variant C never loads degraded data).
- Desync tensors exist for EVERY clip so the real mask can index them; rows of fake clips are zero
  placeholders and are never used.
- Desync view of a real clip: global shift of the clean audio features at load time, or a pre-encoded
  cached desync file when one exists; the type is drawn uniformly [Proposed] (D-13).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from features.cache import cache_path, load_features
from views.degrade import assert_no_heldout
from views.desync import global_shift


class ClipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir, split: str, max_shift_ms,
                 load_degraded: bool, seed: int = 0, pad: str = "roll"):
        if max_shift_ms is None:
            raise ValueError("max_shift_ms is not set: it is an open decision (D-13)")
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no clips in split '{split}'")
        if split == "train":
            assert_no_heldout(self.df)                        # held-out families never in training
        self.root, self.split, self.max_ms = Path(cache_dir), split, max_shift_ms
        self.load_degraded, self.seed, self.pad, self.epoch = load_degraded, seed, pad, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.df)

    def _load(self, view, clip_id):
        return load_features(cache_path(self.root, self.split, view, clip_id))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vc, ac = self._load("clean", r.clip_id)
        item = {"v_clean": vc.float(), "a_clean": ac.float(), "y": int(r.y),
                "clip_id": r.clip_id, "video_id": r.video_id, "category": r.category}
        if self.load_degraded:
            vd, ad = self._load("degraded", r.clip_id)
            item["v_deg"], item["a_deg"] = vd.float(), ad.float()
        if r.y == 0:                                          # desync views: real clips only
            rng = np.random.default_rng([self.seed, self.epoch, int(i)])
            cached = cache_path(self.root, self.split, "desync", r.clip_id)
            if cached.exists() and rng.random() < 0.5:
                vs, as_ = (t.float() for t in load_features(cached))
            else:
                vs, (as_, _) = item["v_clean"], global_shift(item["a_clean"].numpy(), self.max_ms, rng, self.pad)
                as_ = torch.from_numpy(as_)
            item["v_desync"], item["a_desync"] = vs, as_
        else:
            item["v_desync"], item["a_desync"] = torch.zeros_like(item["v_clean"]), torch.zeros_like(item["a_clean"])
        return item


def collate(items):
    out = {k: torch.stack([it[k] for it in items]) for k in items[0] if k not in ("y", "clip_id", "video_id", "category")}
    out["y"] = torch.tensor([it["y"] for it in items])
    for k in ("clip_id", "video_id", "category"):
        out[k] = [it[k] for it in items]
    return out


class BalancedBatchSampler(Sampler):
    """Every batch holds batch_size/2 real and batch_size/2 fake clips.
    Fake clips are visited once per epoch (shuffled); real clips are drawn with replacement
    because they are scarce. An epoch ends when all fake clips have been used.
    This epoch definition is [Proposed]; it is an open decision (D-05)."""

    def __init__(self, labels, batch_size: int, seed: int = 0):
        if batch_size % 2:
            raise ValueError("batch_size must be even for 50/50 batches")
        labels = np.asarray(labels)
        self.real, self.fake = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        if len(self.real) == 0 or len(self.fake) == 0:
            raise ValueError("balanced sampling needs both real and fake clips")
        self.half, self.seed, self.epoch = batch_size // 2, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.fake) / self.half)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        fake = rng.permutation(self.fake)
        for b in range(len(self)):
            f = fake[b * self.half:(b + 1) * self.half]
            if len(f) < self.half:                            # last batch: top up with random fakes
                f = np.r_[f, rng.choice(self.fake, self.half - len(f), replace=False)]
            r = rng.choice(self.real, self.half, replace=len(self.real) < self.half)
            yield [int(i) for i in rng.permutation(np.r_[r, f])]
