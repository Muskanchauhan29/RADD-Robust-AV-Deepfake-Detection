"""Step 8c: evaluate one run on every condition at the SAVED threshold.

    python evaluate.py --run results/D1/seed0

Writes predictions_{condition}.csv, metrics.csv, roc_{condition}.png, confusion_{condition}.png.
There is no code path here that computes or changes a threshold. Held-out versions per condition (D-14) and the
level at which embedding distance is taken (D-09) are open decisions.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from evaluation import metrics as M
from inference import load_run_model, load_view, predict

CONDITIONS = ["clean", "heldout_audio", "heldout_video", "heldout_joint"]   # evidence for H2: the last three
TABLE_10 = {(False, False, False): "C", (True, False, False): "D0", (True, True, False): "D1a",
            (True, False, True): "D1b", (True, True, True): "D1"}


def variant_name(flags: dict) -> str:
    return TABLE_10.get((flags["degraded_cls"], flags["nce_deg_pos"], flags["pair_loss"]), "custom")


def _roc_plot(df, thr, path, title):
    fpr, tpr, _ = roc_curve(df.y, df.score)
    pred = df.score >= thr
    op = ((pred & (df.y == 0)).sum() / max((df.y == 0).sum(), 1), (pred & (df.y == 1)).sum() / max((df.y == 1).sum(), 1))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.scatter(*op, c="red", zorder=3, label="fixed threshold"); ax.set(xlabel="FPR", ylabel="TPR", title=title)
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _confusion_plot(cm, path, title):
    fig, ax = plt.subplots(figsize=(3.5, 3.5)); mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat): ax.text(j, i, str(v), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred real", "pred fake"], yticklabels=["real", "fake"], title=title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def evaluate_run(run_dir, manifest: pd.DataFrame, cache_dir, conditions=CONDITIONS, split="test", device="cpu"):
    run = Path(run_dir)
    thr = json.load(open(run / "threshold.json"))["threshold"]          # saved once; never recomputed here
    model, ck = load_run_model(run, device)
    test = manifest[manifest["split"] == split].reset_index(drop=True)
    vc, ac = load_view(cache_dir, split, "clean", test.clip_id)
    clean = predict(model, vc, ac, device=device)
    real = torch.from_numpy((test.y == 0).to_numpy().copy())
    rows = []
    for cond in conditions:
        o = clean if cond == "clean" else predict(model, *load_view(cache_dir, split, cond, test.clip_id), device=device)
        score = torch.sigmoid(o["logit"]).numpy()
        df = M.add_pred(pd.DataFrame({"clip_id": test.clip_id, "video_id": test.video_id, "category": test.category,
                                      "y": test.y, "score": score}), thr)
        df["mean_c"] = o["c"].mean(1).numpy()
        if cond == "clean":                                              # per-clip drift / embedding distance
            df["drift"] = df["emb_dist"] = np.nan
        else:
            df["drift"] = (o["c"] - clean["c"]).abs().mean(1).numpy()
            df["emb_dist"] = np.mean([(1.0 - (clean[k] * o[k]).sum(-1)).mean(1).numpy() for k in ("ev", "ea")], axis=0)
        df.rename(columns={"score": "p_fake", "pred": "prediction"}).to_csv(run / f"predictions_{cond}.csv", index=False)
        r = M.evaluate_condition(df)
        if cond == "clean":
            r["paired_drift"] = r["emb_cos_dist"] = float("nan")
        else:                                                           # real clips: clean vs degraded views
            r["paired_drift"] = M.paired_drift(clean["c"][real].numpy(), o["c"][real].numpy())
            r["emb_cos_dist"] = float(np.mean([M.embedding_cosine_distance(clean[k][real].numpy(), o[k][real].numpy())
                                               for k in ("ev", "ea")]))      # level and pooling: D-09
        _roc_plot(df, thr, run / f"roc_{cond}.png", f"ROC: {cond}")
        _confusion_plot(M.confusion(df), run / f"confusion_{cond}.png", f"{cond} (thr={thr:.3f})")
        rows.append({"variant": variant_name(ck["config"]["flags"]), "seed": ck["seed"], "condition": cond, **r})
    out = pd.DataFrame(rows); out.to_csv(run / "metrics.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    cfg = torch.load(Path(a.run) / "best.pt", map_location="cpu", weights_only=False)["config"]
    manifest = pd.read_csv(Path(cfg["data"]["manifest_dir"]) / "clips.csv")
    print(evaluate_run(a.run, manifest, cfg["data"]["cache_dir"]).to_string())


if __name__ == "__main__":
    main()