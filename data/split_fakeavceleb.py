"""Step 1: identity-disjoint split of a video-level manifest.

Input CSV columns (one row per source video):
    video_id, identity_id, label (0 real / 1 fake), category, path
Output: same rows plus `split` in {train, val, test}.

All videos of one identity go to the same split, so no identity appears in two splits.
Clipping (2 s windows) comes later and inherits the split of its parent video.
"""
import argparse
import numpy as np
import pandas as pd
import yaml

REQUIRED = ["video_id", "identity_id", "label", "category", "path"]


def split_by_identity(df: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")

    rng = np.random.default_rng(seed)
    ids = np.array(sorted(df["identity_id"].unique()))
    rng.shuffle(ids)

    # Assign whole identities, in shuffled order, to the split furthest below its target
    # share of VIDEOS (not identities), because identities differ in video count.
    n_videos = df.groupby("identity_id").size()
    total = int(n_videos.sum())
    target = {s: r * total for s, r in ratios.items()}
    got = {s: 0 for s in ratios}
    assign = {}
    for i in ids:
        s = max(ratios, key=lambda k: target[k] - got[k])
        assign[i] = s
        got[s] += int(n_videos[i])

    out = df.copy()
    out["split"] = out["identity_id"].map(assign)
    return out


def select_subset(df: pd.DataFrame, n_real: int, n_fake: int, seed: int) -> pd.DataFrame:
    """Small-scale pilot subset [Proposed, supervisor suggestion: about 10 real + 10 fake videos].

    Real videos are taken from DIFFERENT identities where possible, and the fake videos only from those same
    identities (a fake is made from its identity's real video; keeping them together keeps the split identity-disjoint
    and means every test identity has both classes). Fakes are spread over manipulation categories round-robin.
    Raises if the manifest cannot supply the requested numbers.
    """
    rng = np.random.default_rng(seed)
    real, fake = df[df.label == 0], df[df.label == 1]
    ok = sorted(set(real.identity_id) & set(fake.identity_id))
    if len(ok) == 0:
        raise ValueError("no identity has both a real and a fake video")
    rng.shuffle(ok)
    chosen_real = []
    for ident in ok:                                          # one real video per identity first
        chosen_real.append(real[real.identity_id == ident].sample(1, random_state=int(rng.integers(1 << 31))))
        if len(chosen_real) == n_real:
            break
    chosen_real = pd.concat(chosen_real)
    if len(chosen_real) < n_real:
        raise ValueError(f"only {len(chosen_real)} identities have both classes; asked for {n_real} real videos")
    pool = fake[fake.identity_id.isin(chosen_real.identity_id)]
    if len(pool) < n_fake:
        raise ValueError(f"only {len(pool)} fake videos available for the chosen identities; asked for {n_fake}")
    picked, by_cat = [], {c: g.sample(frac=1, random_state=int(rng.integers(1 << 31))) for c, g in pool.groupby("category")}
    while len(picked) < n_fake:                               # round-robin over categories
        for c in sorted(by_cat):
            if len(picked) < n_fake and len(by_cat[c]):
                picked.append(by_cat[c].iloc[:1]); by_cat[c] = by_cat[c].iloc[1:]
    return pd.concat([chosen_real, *picked]).reset_index(drop=True)


def split_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Real / fake video counts and identity counts per split."""
    t = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    t = t.rename(columns={0: "real", 1: "fake"}).reindex(columns=["real", "fake"], fill_value=0)
    t["identities"] = df.groupby("split")["identity_id"].nunique()
    return t


def check_split(df: pd.DataFrame, min_real: int) -> None:
    """Raise if an identity is in two splits or a split has too few real videos."""
    n_splits_per_id = df.groupby("identity_id")["split"].nunique()
    leaked = n_splits_per_id[n_splits_per_id > 1].index.tolist()
    if leaked:
        raise AssertionError(f"identities in more than one split: {leaked[:5]}")
    counts = split_counts(df)
    for s in ["train", "val", "test"]:
        if s not in counts.index or counts.loc[s, "real"] < min_real:
            raise AssertionError(
                f"split '{s}' has too few real videos (< {min_real}); "
                "DR-FPR and L_pair need real clips (D-03). Do not continue.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="video-level CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n_real", type=int, default=None, help="pilot subset: number of real videos")
    ap.add_argument("--n_fake", type=int, default=None, help="pilot subset: number of fake videos")
    ap.add_argument("--min_real", type=int, default=None,
                    help="override the config guard (needed for tiny pilots; the report must then say so)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seed = cfg["seed"] if a.seed is None else a.seed
    raw = pd.read_csv(a.manifest)
    if (a.n_real is None) != (a.n_fake is None):
        raise SystemExit("give both --n_real and --n_fake, or neither")
    if a.n_real is not None:
        raw = select_subset(raw, a.n_real, a.n_fake, seed)
        print(f"PILOT SUBSET: {a.n_real} real + {a.n_fake} fake videos. No scientific claims from this.")
    df = split_by_identity(raw, cfg["data"]["split_ratios"], seed)
    print(split_counts(df).to_string())
    min_real = cfg["data"]["min_real_videos_per_split"] if a.min_real is None else a.min_real
    if a.min_real is not None:
        print(f"WARNING: min real videos per split overridden to {min_real} (config says "
              f"{cfg['data']['min_real_videos_per_split']}); DR-FPR intervals will be unreliable (D-03).")
    check_split(df, min_real)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
