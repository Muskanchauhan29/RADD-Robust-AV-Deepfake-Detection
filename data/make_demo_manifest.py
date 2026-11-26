"""Manifest for the SWAN-DF demo videos (3 swap pairs, downloaded by hand from https://swan-df.github.io/).

Expected folder layout (rename the downloaded files like this):
    demo_videos/pair1/source.mp4  target.mp4  fake_320.mp4  fake_256.mp4  fake_160a.mp4  fake_160b.mp4
    demo_videos/pair2/...         demo_videos/pair3/...
source and target are the two ORIGINAL (real) videos of the pair; every file starting with `fake` is a deepfake.

    python data/make_demo_manifest.py --root demo_videos --out videos_split.csv --train pair1 --val pair2 --test pair3

The split is given by hand: one pair per split, because the real and fake videos of a pair are made from the same people,
so a pair must never be spread over two splits. With so little data this is a DEMO of the pipeline, not an experiment.
The four deepfakes of a pair share the same content at different face resolutions (near-duplicates).
Cite: Korshunov et al., "Vulnerability of Automatic Identity Recognition to Audio-Visual Deepfakes", IJCB 2023.
"""
import argparse
from pathlib import Path
import pandas as pd

REAL_NAMES = {"source", "target"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def build(root, assign: dict) -> pd.DataFrame:
    root, rows = Path(root), []
    for pair, split in assign.items():
        d = root / pair
        if not d.is_dir():
            raise FileNotFoundError(f"missing folder {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT)
        for p in files:
            stem = p.stem.lower()
            if stem in REAL_NAMES:
                label, cat = 0, "real"
            elif stem.startswith("fake"):
                label, cat = 1, "SWAN-DF " + stem
            else:
                raise ValueError(f"cannot tell if {p.name} is real or fake: call it source, target or fake_*")
            rows.append(dict(video_id=f"{pair}_{stem}", identity_id=pair, label=label, category=cat, path=str(p), split=split))
    df = pd.DataFrame(rows)
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        if (sub.label == 0).sum() == 0 or (sub.label == 1).sum() == 0:
            raise ValueError(f"split '{s}' needs at least one real and one fake video; has {len(sub)} videos")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--test", required=True)
    a = ap.parse_args()
    df = build(a.root, {a.train: "train", a.val: "val", a.test: "test"})
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "real", 1: "fake"}).to_string())
    print("DEMO ONLY: a handful of videos, no scientific claims. Dataset: SWAN-DF demo samples.")
    df.to_csv(a.out, index=False); print("wrote", a.out)


if __name__ == "__main__":
    main()