"""Step 11: aggregate runs, paired bootstrap intervals, decision outcome, tables and plots.

    python make_report.py --results results --out results/summary [--delta D --sens_margin S --h1_margin H]

Margins have NO defaults (D-06). Without them intervals are still written, but every label reads "unset (D-06)" and the
H2 verdict is "undetermined". Seeds are combined by averaging the per-seed metric difference inside every bootstrap draw
([Proposed], D-11); per-seed intervals are written too. Everything is computed from the per-run files; nothing by hand.
Not produced: table_h3.csv and table_lavdf.csv (optional controls and LAV-DF need D-04 approval first).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from evaluation import stats as S
from evaluate import CONDITIONS

VARIANTS = ["C", "D0", "D1a", "D1b", "D1"]
HELDOUT = [c for c in CONDITIONS if c != "clean"]
NUM_COLS = ["auc", "eer", "accuracy", "f1", "dr_fpr", "sensitivity", "paired_drift", "emb_cos_dist"]
UNSET = "unset (D-06)"
MIN_REAL_VIDEOS_WARN = 30


# ------------------------------------------------------------------ loading and checks
def load_runs(results, variants, seeds):
    results = Path(results); metrics, preds, missing = [], {}, []
    for v in variants:
        for s in seeds:
            run = results / v / f"seed{s}"
            if not (run / "metrics.csv").exists():
                missing.append(str(run)); continue
            metrics.append(pd.read_csv(run / "metrics.csv"))
            for c in CONDITIONS:
                preds[(v, s, c)] = pd.read_csv(run / f"predictions_{c}.csv")
    if missing:
        raise FileNotFoundError(f"runs without metrics.csv: {missing}")
    return pd.concat(metrics, ignore_index=True), preds


def check_identical_test_clips(preds):
    ref = {}
    for (v, s, c), df in preds.items():
        ids = tuple(df.clip_id)
        if ref.setdefault(c, ids) != ids:
            raise ValueError(f"variant {v} seed {s} condition {c} was not evaluated on the identical test clips")


def with_heldout_all(preds):
    out = dict(preds)
    for (v, s, c) in list(preds):
        if c == "clean" or (v, s, "heldout_all") in out:
            continue
        out[(v, s, "heldout_all")] = pd.concat([preds[(v, s, h)] for h in HELDOUT], ignore_index=True)
    return out


# ------------------------------------------------------------------ per-video sufficient statistics and bootstrap
def video_stats(df, metric):
    """(video ids, numerator, denominator) per video for dr_fpr, sensitivity or drift."""
    if metric == "dr_fpr":
        d = df[df.y == 0]; num = d.groupby("video_id").prediction.sum(); den = d.groupby("video_id").size()
    elif metric == "sensitivity":
        d = df[df.y == 1]; num = d.groupby("video_id").prediction.sum(); den = d.groupby("video_id").size()
    elif metric == "drift":
        d = df[df.y == 0]; num = d.groupby("video_id").drift.sum(); den = d.groupby("video_id").size()
    else:
        raise ValueError(metric)
    vids = sorted(df.video_id.unique())
    return vids, num.reindex(vids, fill_value=0).to_numpy(float), den.reindex(vids, fill_value=0).to_numpy(float)


def bootstrap(frames_a, frames_b, metric, n_boot=1000, seed=0, max_nan_frac=0.05):
    """95% interval for mean over seeds of [metric(a) - metric(b)] over resampled VIDEOS.
    frames_*: lists with one frame per seed (same videos everywhere). frames_b=None gives the interval of metric(a) alone."""
    sa = [video_stats(f, metric) for f in frames_a]
    sb = [video_stats(f, metric) for f in frames_b] if frames_b is not None else None
    vids = sa[0][0]
    assert all(x[0] == vids for x in sa + (sb or [])), "frames must contain the same videos"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(n_boot, len(vids)))
    def rate(stats):
        num, den = np.stack([x[1] for x in stats]), np.stack([x[2] for x in stats])             # [seeds, videos]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array([num[k][idx].sum(1) / den[k][idx].sum(1) for k in range(len(stats))]).mean(0)
    d = rate(sa) - (rate(sb) if sb else 0.0)
    if np.isnan(d).mean() > max_nan_frac:
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.nanpercentile(d, [2.5, 97.5]))


def point(frames, metric):
    r = []
    for f in frames:
        _, num, den = video_stats(f, metric); r.append(num.sum() / den.sum() if den.sum() else np.nan)
    return float(np.mean(r))


# ------------------------------------------------------------------ comparisons
DECISION = [("D1b", "D0"), ("D1", "D1a")]
DESCRIPTIVE = [("D1a", "D0"), ("D1", "D1b"), ("D0", "C")]
CONDS_CI = HELDOUT + ["heldout_all"]


def label(ci, metric, delta, sens_margin):
    if np.isnan(ci[0]):
        return "inconclusive (undefined interval)"
    if metric == "sensitivity":
        return UNSET if sens_margin is None else ("within margin" if S.sens_ok(ci, sens_margin) else "loss beyond margin")
    return UNSET if delta is None else S.classify(ci, delta)


def build_ci_table(preds, variants, seeds, n_boot, delta, sens_margin, h1_margin):
    rows = []
    def add(kind, a, b, metric, cond, fa, fb, per_seed=True):
        for tag, sel in (("pooled", slice(None)),) + ((tuple((s, slice(i, i + 1)) for i, s in enumerate(seeds))) if per_seed else ()):
            la, lb = fa[sel], (fb[sel] if fb is not None else None)
            ci = bootstrap(la, lb, metric, n_boot)
            diff = point(la, metric) - (point(lb, metric) if lb is not None else 0)
            rows.append(dict(kind=kind, comparison=f"{a}-{b}", metric=metric, condition=cond,
                             seed=tag if tag == "pooled" else tag, diff=diff, ci_low=ci[0], ci_high=ci[1],
                             label=label(ci, metric, delta, sens_margin)))
    for kind, pairs in (("decision", DECISION), ("descriptive", DESCRIPTIVE)):
        for a, b in pairs:
            if a not in variants or b not in variants:
                continue
            for cond in CONDS_CI:
                for metric in ("dr_fpr", "sensitivity", "drift"):
                    fa = [preds[(a, s, cond)] for s in seeds]; fb = [preds[(b, s, cond)] for s in seeds]
                    add(kind, a, b, metric, cond, fa, fb)
    if "C" in variants:                                   # H1: FPR(degraded) - FPR(clean) for variant C
        for cond in CONDS_CI:
            fa = [preds[("C", s, cond)] for s in seeds]; fb = [preds[("C", s, "clean")] for s in seeds]
            add("h1", "C_degraded", "C_clean", "dr_fpr", cond, fa, fb)
            for r in rows[-(len(seeds) + 1):]:
                ci = (r["ci_low"], r["ci_high"])
                r["label"] = UNSET if h1_margin is None else ("above margin" if ci[0] > h1_margin else "not above margin")
    return pd.DataFrame(rows)


def h2_verdict(ci, delta, sens_margin):
    """Outcome of the decision rule on the pooled heldout_all intervals (condition choice: [Proposed])."""
    if delta is None or sens_margin is None:
        return {"outcome": "undetermined: margins unset (D-06)"}
    g = lambda a, b, m: tuple(ci[(ci.kind == "decision") & (ci.comparison == f"{a}-{b}") & (ci.metric == m)
                              & (ci.condition == "heldout_all") & (ci.seed == "pooled")][["ci_low", "ci_high"]].iloc[0])
    d1b, d1, s1b, s1 = g("D1b", "D0", "dr_fpr"), g("D1", "D1a", "dr_fpr"), g("D1b", "D0", "sensitivity"), g("D1", "D1a", "sensitivity")
    sens_ok = S.sens_ok(s1b, sens_margin) and S.sens_ok(s1, sens_margin)
    detail = {"D1b-D0 dr_fpr": d1b, "D1-D1a dr_fpr": d1, "sens D1b-D0": s1b, "sens D1-D1a": s1}
    if S.improved(d1b) and S.improved(d1) and sens_ok:
        out = "supported"
    elif S.equivalent(d1, delta) and sens_ok:
        out = "redundant (D1a equivalent to D1)"
    elif S.worse(d1b) or S.worse(d1) or not sens_ok:
        out = "not supported"
    else:
        out = "inconclusive"
    return {"outcome": out, "intervals": detail}


# ------------------------------------------------------------------ plots
def plot_all(out, metrics, preds, variants, seeds, ci, histories):
    colors = {v: f"C{i}" for i, v in enumerate(variants)}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for v in variants:
        h = pd.concat(histories[v]).groupby("epoch").mean(numeric_only=True)
        ax[0].plot(h.index, h.total_loss, c=colors[v], label=v); ax[1].plot(h.index, h.val_auc, c=colors[v], label=v)
    ax[0].set(title="training loss", xlabel="epoch"); ax[1].set(title="clean validation AUC", xlabel="epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=120); plt.close(fig)

    grid = np.linspace(0, 1, 101)
    for c in CONDITIONS:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for v in variants:
            t = []
            for s in seeds:
                d = preds[(v, s, c)]; f, tp, _ = roc_curve(d.y, d.p_fake); t.append(np.interp(grid, f, tp))
            ax.plot(grid, np.mean(t, 0), c=colors[v], label=v)
        ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set(title=f"ROC (mean over seeds): {c}", xlabel="FPR", ylabel="TPR"); ax.legend()
        fig.tight_layout(); fig.savefig(out / f"roc_overlay_{c}.png", dpi=120); plt.close(fig)

    ms = metrics.groupby(["variant", "condition"])[NUM_COLS].agg(["mean", "std"])
    def single_ci(v, cond):
        return bootstrap([preds[(v, s, cond)] for s in seeds], None, "dr_fpr", 1000)
    fig, ax = plt.subplots(figsize=(7, 3.5)); w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        y = [point([preds[(v, s, c)] for s in seeds], "dr_fpr") for c in CONDS_CI]
        e = np.array([single_ci(v, c) for c in CONDS_CI]).T
        ax.bar(np.arange(len(CONDS_CI)) + i * w, y, w, yerr=[np.array(y) - e[0], e[1] - np.array(y)], label=v, color=colors[v], capsize=2)
    ax.set_xticks(np.arange(len(CONDS_CI)) + 0.4 - w / 2, CONDS_CI); ax.set(ylabel="DR-FPR (lower is better)", title="DR-FPR with bootstrap intervals"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "comparison_drfpr.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for v in variants:
        fr = [preds[(v, s, "heldout_all")] for s in seeds]
        ax.scatter(point(fr, "dr_fpr"), point(fr, "sensitivity"), c=colors[v], label=v)
    ax.set(xlabel="DR-FPR", ylabel="sensitivity", title="held-out: sensitivity vs DR-FPR"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "comparison_sens_vs_drfpr.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for k, col in enumerate(("paired_drift", "emb_cos_dist")):
        vals = [ms.loc[(v, HELDOUT), (col, "mean")].mean() for v in variants]
        ax[k].bar(variants, vals, color=[colors[v] for v in variants]); ax[k].set_title(col + " (mean of held-out conditions)")
    fig.tight_layout(); fig.savefig(out / "comparison_drift.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for k, col in enumerate(("auc", "eer")):
        for i, v in enumerate(variants):
            ax[k].bar(np.arange(len(CONDITIONS)) + i * w, [ms.loc[(v, c), (col, "mean")] for c in CONDITIONS], w,
                      yerr=[ms.loc[(v, c), (col, "std")] for c in CONDITIONS], color=colors[v], label=v, capsize=2)
        ax[k].set_xticks(np.arange(len(CONDITIONS)) + 0.4 - w / 2, CONDITIONS, rotation=20); ax[k].set_title(col)
    ax[0].legend(); fig.tight_layout(); fig.savefig(out / "comparison_auc_eer.png", dpi=120); plt.close(fig)


# ------------------------------------------------------------------ main
def make_report(results, out, variants=VARIANTS, seeds=(0, 1, 2), n_boot=1000, delta=None, sens_margin=None, h1_margin=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    metrics, preds = load_runs(results, variants, seeds)
    check_identical_test_clips(preds)
    metrics.to_csv(out / "metrics_all.csv", index=False)
    ms = metrics.groupby(["variant", "condition"])[NUM_COLS].agg(["mean", "std"])
    ms.columns = [f"{a}_{b}" for a, b in ms.columns]; ms.reset_index().to_csv(out / "metrics_mean_std.csv", index=False)
    preds = with_heldout_all(preds)
    ci = build_ci_table(preds, variants, list(seeds), n_boot, delta, sens_margin, h1_margin)
    ci.to_csv(out / "bootstrap_ci.csv", index=False)
    verdict = h2_verdict(ci, delta, sens_margin)
    n_real = int(preds[(variants[0], seeds[0], "clean")].query("y == 0").video_id.nunique())
    verdict["n_real_test_videos"] = n_real
    if n_real < MIN_REAL_VIDEOS_WARN:                                     # [Proposed] warning level, not a statistical rule
        verdict["warning"] = (f"only {n_real} real test videos: intervals are unreliable, so any outcome here (including "
                              f"'supported' or 'redundant') is NOT evidence. Report it as a pilot.")
        print("WARNING:", verdict["warning"])
    json.dump(verdict, open(out / "h2_verdict.json", "w"), indent=2, default=list)
    ms2 = ms.reset_index()
    ms2[ms2.variant == "C"].to_csv(out / "table_h1.csv", index=False)
    ms2[ms2.condition.isin(HELDOUT)][["variant", "condition"] + [c for c in ms2.columns if c.startswith(("dr_fpr", "sensitivity", "paired_drift", "emb_cos_dist"))]] \
        .to_csv(out / "table_h2.csv", index=False)
    histories = {v: [pd.read_csv(Path(results) / v / f"seed{s}" / "training_history.csv") for s in seeds] for v in variants}
    plot_all(out, metrics, preds, variants, list(seeds), ci, histories)
    return ci, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)                     # D-11
    ap.add_argument("--delta", type=float, default=None); ap.add_argument("--sens_margin", type=float, default=None)
    ap.add_argument("--h1_margin", type=float, default=None)
    a = ap.parse_args()
    ci, verdict = make_report(a.results, a.out, n_boot=a.n_boot, delta=a.delta, sens_margin=a.sens_margin, h1_margin=a.h1_margin)
    print("H2 outcome:", verdict["outcome"])


if __name__ == "__main__":
    main()
