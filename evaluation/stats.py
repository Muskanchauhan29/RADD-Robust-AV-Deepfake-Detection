"""Step 6: paired bootstrap over videos and the decision helpers (Guide §12.4, corrected rule).
Every difference is variant MINUS reference. Lower DR-FPR / drift is better, so an improvement
needs the whole interval below zero. An interval that merely contains zero is NOT equivalence.
delta (equivalence margin) and the sensitivity margin are unset (D-06): always passed in, no default.
"""
import numpy as np
import pandas as pd


def bootstrap_delta(da, db, metric, n_boot=1000, seed=0, max_nan_frac=0.05):
    """95% interval for metric(da) - metric(db) over resampled VIDEOS (same videos in both frames).
    n_boot and how seeds are combined: D-11. max_nan_frac is a [Proposed] guard: if more than this
    share of resamples gives an undefined metric (e.g. no real clip drawn) the interval is (nan, nan)."""
    ga, gb = dict(tuple(da.groupby("video_id"))), dict(tuple(db.groupby("video_id")))
    if set(ga) != set(gb):
        raise ValueError("da and db must contain the same video_ids (paired comparison)")
    rng = np.random.default_rng(seed)
    vids = np.array(sorted(ga)); d = []
    for _ in range(n_boot):
        s = rng.choice(vids, len(vids), replace=True)
        d.append(metric(pd.concat([ga[v] for v in s])) - metric(pd.concat([gb[v] for v in s])))
    d = np.array(d, dtype=float)
    if np.isnan(d).mean() > max_nan_frac:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in np.nanpercentile(d, [2.5, 97.5]))      # (lower, upper)


def improved(ci):                 # lower is better: the whole interval must be below zero
    return ci[1] < 0


def worse(ci):                    # whole interval above zero
    return ci[0] > 0


def equivalent(ci, delta):        # inside the predefined margin (-delta, +delta); containing 0 is not enough
    return -delta < ci[0] and ci[1] < delta


def sens_ok(ci, margin):          # sensitivity: higher is better; a loss beyond the margin fails
    return ci[0] > -margin


def classify(ci, delta):
    if improved(ci): return "improved"
    if worse(ci): return "worse"
    if equivalent(ci, delta): return "equivalent"
    return "inconclusive"         # also covers (nan, nan)
