"""Synthetic video-level manifest for testing the pipeline. NOT FakeAVCeleb; no real data.

Mimics only the structure: few real videos, many fake ones, several videos per identity.
Replace with the real FakeAVCeleb metadata once access is confirmed (D-03).
"""
import argparse
import numpy as np
import pandas as pd

CATEGORIES = ["FakeVideo-FakeAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio"]


def make(n_identities=100, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_identities):
        ident = f"id{i:04d}"
        for r in range(int(rng.integers(1, 4))):                # 1-3 real videos
            rows.append((f"{ident}_real{r}", ident, 0, "real", f"synthetic/{ident}/real{r}.mp4"))
        for f in range(int(rng.integers(5, 15))):               # 5-14 fake videos
            cat = CATEGORIES[int(rng.integers(len(CATEGORIES)))]
            rows.append((f"{ident}_fake{f}", ident, 1, cat, f"synthetic/{ident}/fake{f}.mp4"))
    return pd.DataFrame(rows, columns=["video_id", "identity_id", "label", "category", "path"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_manifest.csv")
    ap.add_argument("--n_identities", type=int, default=100)
    a = ap.parse_args()
    make(a.n_identities).to_csv(a.out, index=False)
    print("wrote", a.out)
