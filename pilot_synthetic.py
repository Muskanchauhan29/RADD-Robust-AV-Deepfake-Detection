"""Step 8d: end-to-end plumbing check on SYNTHETIC data.

    python pilot_synthetic.py --out pilot_out --variants D1

Builds a fake manifest, fake cached features and runs split -> train -> threshold -> evaluate, then checks that every
per-run file of Table 14 exists. The data are random numbers with an artificial class signal: the metrics it prints
say NOTHING about the method. The real pilot (one seed, real clips and encoders) has not been run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate as EV
import threshold as TH
import train as TR
from config import load_config
from data.make_synthetic_manifest import make
from data.split_fakeavceleb import check_split, split_by_identity, split_counts
from features import cache

T, DV, DA = 50, 12, 10
PER_RUN_FILES = (["config.yaml", "train.log", "training_history.csv", "best.pt", "threshold.json", "metrics.csv"]
                 + [f"{p}_{c}.{e}" for c in EV.CONDITIONS for p, e in (("predictions", "csv"), ("roc", "png"), ("confusion", "png"))])


def build_world(out: Path, n_identities=30, seed=0, min_real=3):
    videos = split_by_identity(make(n_identities, seed), {"train": .70, "val": .15, "test": .15}, seed)
    check_split(videos, min_real=min_real)
    print(split_counts(videos).to_string())
    clips = videos.rename(columns={"label": "y"}).assign(clip_id=lambda d: d.video_id + "_c0", degradation="noise")
    rng = np.random.default_rng(seed); cache_dir = out / "cache"
    for r in clips.itertuples():
        base = rng.standard_normal((T, DV)) + 1.5 * r.y                      # artificial class signal
        audio = rng.standard_normal((T, DA))
        views = ["clean", "degraded"] + (EV.CONDITIONS[1:] if r.split == "test" else [])
        for view in views:
            v = base + (0 if view == "clean" else 0.3) * rng.standard_normal((T, DV))
            cache.save_features(cache.cache_path(cache_dir, r.split, view, r.clip_id), torch.from_numpy(v), torch.from_numpy(audio))
    (out / "manifests").mkdir(parents=True, exist_ok=True); clips.to_csv(out / "manifests" / "clips.csv", index=False)
    return clips, cache_dir


def run_variant(variant, clips, cache_dir, out: Path, epochs, seed=0):
    cfg = load_config(variant); cfg.data.cache_dir = str(cache_dir)
    cfg.model.d, cfg.model.layers, cfg.model.proj_dim = 32, 1, 16
    cfg.train.batch_size, cfg.train.max_epochs, cfg.train.lr = 8, epochs, 1e-3
    cfg.desync.max_shift_ms = 200            # test value only: the real value is an open decision (D-13)
    run = TR.run_training(cfg, clips, out / "results" / variant / f"seed{seed}", seed, "cpu")
    TH.write_threshold(run, clips, cache_dir)
    metrics = EV.evaluate_run(run, clips, cache_dir)
    missing = [f for f in PER_RUN_FILES if not (run / f).exists()]
    assert not missing, f"missing per-run files: {missing}"
    return run, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot_out"); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=["D1"])
    a = ap.parse_args(); out = Path(a.out)
    print("=== SYNTHETIC DATA: plumbing check only, metrics are meaningless ===")
    clips, cache_dir = build_world(out)
    for v in a.variants:
        run, m = run_variant(v, clips, cache_dir, out, a.epochs)
        print(f"\n{v}: all {len(PER_RUN_FILES)} per-run files present in {run}")
        print(m[["variant", "seed", "condition", "auc", "dr_fpr", "sensitivity"]].to_string(index=False))


if __name__ == "__main__":
    main()
