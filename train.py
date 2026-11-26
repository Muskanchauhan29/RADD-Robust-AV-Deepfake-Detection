"""Step 8b: single training script; the variant is chosen by the config's flags only.

    python train.py --config configs/D1.yaml --seed 0 --out results/D1/seed0

Writes per run: config.yaml, train.log, training_history.csv, best.pt   (threshold.json: threshold.py)
Early stopping uses the clean-validation AUC (metric choice: D-10). Nothing here has been run on real data.
"""
import argparse
import logging
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import datasets as DS
from config import load_config
from features.cache import cache_path, load_features
from losses import cls_loss
from model.heads import RACL
from train_step import training_step

HISTORY_COLS = ["epoch", "L_cls", "L_NCE", "L_pair", "total_loss", "val_auc", "val_loss", "lr", "seconds"]


def ns_to_dict(x):
    return {k: ns_to_dict(v) for k, v in vars(x).items()} if isinstance(x, SimpleNamespace) else x


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def make_logger(path):
    log = logging.getLogger(f"racl.{path}"); log.setLevel(logging.INFO); log.handlers.clear()
    for h in (logging.FileHandler(path), logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s")); log.addHandler(h)
    return log


class EarlyStopper:
    """Stops when the monitored value (higher is better) has not improved for `patience` epochs."""
    def __init__(self, patience):
        self.patience, self.best, self.bad = patience, -np.inf, 0
    def update(self, value) -> tuple[bool, bool]:          # returns (improved, stop)
        if value > self.best:
            self.best, self.bad = value, 0; return True, False
        self.bad += 1; return False, self.bad >= self.patience


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); logits, ys = [], []
    for b in loader:
        logits.append(model(b["v_clean"].to(device), b["a_clean"].to(device))["logit"].cpu()); ys.append(b["y"])
    z, y = torch.cat(logits), torch.cat(ys)
    if len(torch.unique(y)) < 2:
        raise ValueError("validation split needs both real and fake clips")
    return float(roc_auc_score(y.numpy(), torch.sigmoid(z).numpy())), float(cls_loss(z, y))


def run_training(cfg, manifest: pd.DataFrame, out_dir, seed: int, device=None) -> Path:
    if cfg.desync.max_shift_ms is None:
        raise ValueError("desync.max_shift_ms is not set (decision D-13): refusing to train")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "train.log")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    yaml.safe_dump({"config": ns_to_dict(cfg), "seed": seed, "git_commit": git_commit()},
                   open(out / "config.yaml", "w"), sort_keys=False)

    f, t = cfg.flags, cfg.train
    need_deg = f.degraded_cls or f.nce_deg_pos or f.pair_loss
    cache = cfg.data.cache_dir
    train_ds = DS.ClipDataset(manifest, cache, "train", cfg.desync.max_shift_ms, need_deg, seed)
    val_ds = DS.ClipDataset(manifest, cache, "val", cfg.desync.max_shift_ms, False, seed)
    sampler = DS.BalancedBatchSampler(train_ds.df.y.to_numpy(), t.batch_size, seed)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=DS.collate)
    val_dl = DataLoader(val_ds, batch_size=t.batch_size, collate_fn=DS.collate)

    first = train_ds.df.iloc[0].clip_id
    dv, da = (x.shape[1] for x in load_features(cache_path(cache, "train", "clean", first)))
    m = cfg.model
    model = RACL(dv, da, d=m.d, layers=m.layers, heads=m.heads, w=m.window,
                 bias_range=m.bias_range, proj_dim=m.proj_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    stopper, rows = EarlyStopper(t.patience), []
    log.info(f"variant flags={ns_to_dict(f)} seed={seed} device={device} train_clips={len(train_ds)}")

    for epoch in range(1, t.max_epochs + 1):
        t0 = time.time(); model.train(); train_ds.set_epoch(epoch); sampler.set_epoch(epoch)
        acc = {"loss": [], "cls": [], "nce": [], "pair": []}
        for b in train_dl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss, parts = training_step(model, b, cfg)
            opt.zero_grad(); loss.backward(); opt.step()
            acc["loss"].append(loss.item())
            for k in ("cls", "nce", "pair"):
                acc[k].append(np.nan if parts[k] is None else parts[k])
        auc, vloss = validate(model, val_dl, device)
        mean = lambda k: float(np.nanmean(acc[k])) if not np.all(np.isnan(acc[k])) else float("nan")
        rows.append([epoch, mean("cls"), mean("nce"), mean("pair"), float(np.mean(acc["loss"])),
                     auc, vloss, opt.param_groups[0]["lr"], time.time() - t0])
        pd.DataFrame(rows, columns=HISTORY_COLS).to_csv(out / "training_history.csv", index=False)
        improved, stop = stopper.update(auc)
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": auc,
                        "dims": {"dv": dv, "da": da}, "config": ns_to_dict(cfg), "seed": seed}, out / "best.pt")
        log.info(f"epoch {epoch} loss={rows[-1][4]:.4f} val_auc={auc:.4f} val_loss={vloss:.4f}"
                 f"{' *' if improved else ''}")
        if stop:
            log.info(f"early stop: no val_auc improvement for {t.patience} epochs"); break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    manifest = pd.read_csv(Path(cfg.data.manifest_dir) / "clips.csv")
    run_training(cfg, manifest, a.out, a.seed)


if __name__ == "__main__":
    main()
