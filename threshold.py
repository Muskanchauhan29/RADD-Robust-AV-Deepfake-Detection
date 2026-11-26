"""Step 8c: EER threshold from CLEAN VALIDATION only, saved once per run.

    python threshold.py --run results/D1/seed0

The threshold is never recomputed or changed for degraded, held-out or test conditions [Paper].
An existing threshold.json is never overwritten.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from evaluation.metrics import eer_threshold
from inference import load_run_model, load_view, predict


def write_threshold(run_dir, manifest: pd.DataFrame, cache_dir, device="cpu") -> dict:
    run = Path(run_dir); path = run / "threshold.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists: the threshold is saved once and never changed")
    val = manifest[manifest["split"] == "val"].reset_index(drop=True)
    model, ck = load_run_model(run, device)
    v, a = load_view(cache_dir, "val", "clean", val.clip_id)
    p_fake = torch.sigmoid(predict(model, v, a, device=device)["logit"]).numpy()
    thr, eer = eer_threshold(val.y.to_numpy(), p_fake)
    res = {"threshold": thr, "val_eer": eer, "split": "val", "view": "clean",
           "n_real": int((val.y == 0).sum()), "n_fake": int((val.y == 1).sum()),
           "checkpoint_epoch": int(ck["epoch"])}
    json.dump(res, open(path, "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    a = ap.parse_args()
    ck = torch.load(Path(a.run) / "best.pt", map_location="cpu", weights_only=False)["config"]
    manifest = pd.read_csv(Path(ck["data"]["manifest_dir"]) / "clips.csv")
    print(write_threshold(a.run, manifest, ck["data"]["cache_dir"]))


if __name__ == "__main__":
    main()
