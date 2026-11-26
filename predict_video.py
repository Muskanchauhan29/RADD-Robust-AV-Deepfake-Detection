"""Run a trained RACL run on one video: cut 2 s clips, extract features, score, compare with the saved threshold.

Uses exactly what evaluate.py uses: p_fake = sigmoid(logit), flagged fake if p_fake >= threshold from threshold.json
(the threshold is read, never recomputed). The video-level verdict below is a DEMO rule [Proposed]: the mean p_fake over
the video's 2 s clips against the same threshold. The paper and the evaluation work per clip.
This is only as good as the run behind it: a pilot trained on a handful of videos is NOT a reliable detector.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import make_clips as MC
from features.extract import extract_clip
from inference import load_run_model, predict


def predict_video(path, run_dir, video_encoder, audio_encoder, size=96, max_clips=None, device="cpu") -> dict:
    run = Path(run_dir)
    thr = json.load(open(run / "threshold.json"))["threshold"]
    frames, wav = MC.read_frames(path), MC.read_audio(path)
    n = MC.n_windows(len(frames), len(wav), max_clips)
    if n == 0:
        raise ValueError("the video needs at least 2 seconds of picture and sound")
    vs, as_ = [], []
    for k in range(n):
        f = MC.center_crop(frames[k * 50:(k + 1) * 50], size)        # centre crop = placeholder for the face crop (D-02)
        v, a = extract_clip(f, wav[k * MC.N_SAMP:(k + 1) * MC.N_SAMP], video_encoder, audio_encoder)
        vs.append(v.float()); as_.append(a.float())
    model, _ = load_run_model(run, device)
    p = torch.sigmoid(predict(model, torch.stack(vs), torch.stack(as_), device=device)["logit"]).numpy()
    clips = pd.DataFrame({"start_s": np.arange(n) * MC.CLIP_S, "p_fake": p, "flagged_fake": (p >= thr).astype(int)})
    mean_p = float(p.mean())
    return {"threshold": float(thr), "clips": clips, "mean_p_fake": mean_p, "verdict": "FAKE" if mean_p >= thr else "REAL"}