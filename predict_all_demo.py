"""Score every video of videos_split.csv with one trained run and write a table you can show.

    python predict_all_demo.py --run results/D1/seed0 --manifest videos_split.csv --out demo_predictions.csv

Columns: video, split, truth, verdict, correct, mean_p_fake, threshold, n_clips.
`correct` on a TRAIN or VAL video only shows the model remembers it; only TEST rows say anything about unseen videos, and
even those are just a few videos. Verdict rule [Proposed]: mean p_fake over the 2 s clips against the saved threshold.
"""
import argparse
from pathlib import Path

import pandas as pd

from predict_video import predict_video


def predict_all(manifest: pd.DataFrame, run_dir, video_encoder, audio_encoder, root=".", size=96) -> pd.DataFrame:
    rows = []
    for r in manifest.itertuples():
        res = predict_video(Path(root) / r.path, run_dir, video_encoder, audio_encoder, size=size)
        truth = "fake" if r.label else "real"
        rows.append(dict(video=f"{Path(r.path).parent.name}/{Path(r.path).name}", split=r.split, truth=truth, verdict=res["verdict"].lower(),
                         correct=int(res["verdict"].lower() == truth), mean_p_fake=round(res["mean_p_fake"], 3),
                         threshold=round(res["threshold"], 3), n_clips=len(res["clips"])))
    return pd.DataFrame(rows).sort_values(["split", "truth", "video"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--manifest", default="videos_split.csv")
    ap.add_argument("--out", default="demo_predictions.csv"); ap.add_argument("--root", default=".")
    a = ap.parse_args()
    from features.real_encoders import load_audio_encoder, load_video_encoder
    df = predict_all(pd.read_csv(a.manifest), a.run, load_video_encoder(), load_audio_encoder(), a.root)
    df.to_csv(a.out, index=False); print(df.to_string(index=False))
    for s, g in df.groupby("split"):
        print(f"{s}: {int(g.correct.sum())} of {len(g)} videos correct" + ("   <- the model saw these in training" if s != "test" else "   <- unseen videos"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
