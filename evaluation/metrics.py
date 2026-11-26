"""Step 6: threshold and metrics. Positive class = fake (y = 1); score = P(fake).
The threshold comes from clean validation only and is never changed afterwards [Paper].
df columns: clip_id, video_id, y, score, pred (pred = score >= threshold); category optional.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, f1_score


def eer_threshold(y, score):
    """Threshold at the equal-error-rate point of the ROC; returns (threshold, eer)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise ValueError("EER needs both real and fake clips")
    fpr, tpr, thr = roc_curve(y, score)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float(thr[i]), float((fpr[i] + 1 - tpr[i]) / 2)


def add_pred(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy(); out["pred"] = (out["score"] >= threshold).astype(int); return out


def dr_fpr(df):
    """Fraction of REAL clips predicted fake. Pass the held-out degraded real clips for DR-FPR."""
    r = df[df.y == 0]
    return float((r.pred == 1).mean()) if len(r) else float("nan")


def sensitivity(df):
    f = df[df.y == 1]
    return float((f.pred == 1).mean()) if len(f) else float("nan")


def paired_drift(c_clean, c_deg):
    """Mean over real clips and frames of |c_t(degraded) - c_t(clean)|. Inputs [N, T]."""
    return float(np.abs(np.asarray(c_deg) - np.asarray(c_clean)).mean())


def embedding_cosine_distance(e_clean, e_deg):
    """Per-frame cosine distance between unit-normalised embeddings [N, T, D], averaged.
    Level and pooling are open (D-09); the per-frame mean is used here."""
    e_clean, e_deg = np.asarray(e_clean), np.asarray(e_deg)
    return float((1.0 - (e_clean * e_deg).sum(-1)).mean())


def confusion(df):
    y, p = df.y.to_numpy(), df.pred.to_numpy()
    return {"tn": int(((y == 0) & (p == 0)).sum()), "fp": int(((y == 0) & (p == 1)).sum()),
            "fn": int(((y == 1) & (p == 0)).sum()), "tp": int(((y == 1) & (p == 1)).sum())}


def evaluate_condition(df) -> dict:
    """All per-condition metrics at the fixed threshold (df already has `pred`)."""
    out = {"n": len(df), **confusion(df)}
    two = df.y.nunique() == 2
    out["auc"] = float(roc_auc_score(df.y, df.score)) if two else float("nan")
    out["eer"] = eer_threshold(df.y, df.score)[1] if two else float("nan")
    out["accuracy"] = float((df.pred == df.y).mean())
    out["f1"] = float(f1_score(df.y, df.pred, zero_division=0))
    out["dr_fpr"], out["sensitivity"] = dr_fpr(df), sensitivity(df)
    return out


def per_category(df) -> pd.DataFrame:
    """The metrics split by manipulation category (reported even when weak)."""
    return pd.DataFrame({k: evaluate_condition(g) for k, g in df.groupby("category")}).T
