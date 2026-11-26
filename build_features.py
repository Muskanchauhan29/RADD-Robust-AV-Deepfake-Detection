"""Step 3 for the pilot: clips -> degraded views -> encoder features -> cache + manifest.

    python build_features.py --clips data/clips/clips.csv --encoders real     (needs transformers + torchvision, internet once)
    python build_features.py --clips data/clips/clips.csv --encoders standin   (no downloads; tiny test encoders)

Views written per clip (features/cache/{split}/{view}/{clip_id}.pt):
    clean, degraded                    for every split
    heldout_audio / _video / _joint    for the TEST split only
Writes manifests/clips.csv (the file train.py, threshold.py and evaluate.py read).

PILOT PLACEHOLDERS (all [Proposed], the real values are open decisions, D-14):
    training degradation : one audio family (noise, SNR 5-25 dB, from the paper) + one video family (gaussian blur, illustrative)
                           (MP3, H.264 are skipped here because they need video re-encoding)
    held-out audio       : reverb with a SYNTHETIC room impulse response (real RIR source undecided)
    held-out video       : linear motion blur
    held-out joint       : both of the above (the real held-out joint condition is a re-encoding chain, D-14)
Every random draw is seeded from the clip_id, so the cache is reproducible.
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features.cache import cache_path
from features.extract import extract_and_save
from views import degrade as D
from views.codecs import SR

RT60_S = 0.5                    # [pilot placeholder, D-14]
MOTION_LENGTH, MOTION_ANGLE = 9, 0.0     # [pilot placeholder, D-14]
TRAIN_DEGRADATION = "noise+gblur"


def clip_rng(clip_id: str, tag: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{clip_id}/{tag}".encode()))


def make_views(clip_id: str, frames: np.ndarray, wav: np.ndarray, split: str) -> dict:
    """{view: (frames, wav)} for the views this split needs. Inputs are never modified."""
    r = clip_rng(clip_id, "train-degradation")
    snr = D.sample_train_params("audio", "noise", r)["snr_db"]
    sigma = D.sample_train_params("video", "gblur", r)["sigma"]
    views = {"clean": (frames, wav),
             "degraded": (D.gaussian_blur(frames, sigma), D.add_noise_snr(wav, snr, r).astype(np.float32))}
    if split == "test":
        rh = clip_rng(clip_id, "heldout")
        rev = D.reverb(wav, D.synthetic_rir(SR, RT60_S, rh)).astype(np.float32)
        mb = D.motion_blur(frames, MOTION_LENGTH, MOTION_ANGLE)
        views.update(heldout_audio=(frames, rev), heldout_video=(mb, wav), heldout_joint=(mb, rev))
    return views


def build(clips: pd.DataFrame, cache_dir, video_encoder, audio_encoder, skip_existing=True) -> int:
    n = 0
    for r in clips.itertuples():
        d = np.load(r.path); frames, wav = d["frames"], d["wav"]
        for view, (f, w) in make_views(r.clip_id, frames, wav, r.split).items():
            p = cache_path(cache_dir, r.split, view, r.clip_id)
            if skip_existing and p.exists():
                continue
            extract_and_save(p, f, w, video_encoder, audio_encoder); n += 1
        if (n and n % 50 == 0):
            print(f"  {n} views done", flush=True)
    return n


def write_manifest(clips: pd.DataFrame, out) -> pd.DataFrame:
    m = clips[["clip_id", "video_id", "identity_id", "y", "category", "split"]].assign(degradation=TRAIN_DEGRADATION)
    Path(out).parent.mkdir(parents=True, exist_ok=True); m.to_csv(out, index=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--cache", default="features/cache")
    ap.add_argument("--out_manifest", default="manifests/clips.csv")
    ap.add_argument("--encoders", choices=["real", "standin"], required=True); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.encoders == "real":
        from features.real_encoders import load_audio_encoder, load_video_encoder
        venc, aenc = load_video_encoder(a.device), load_audio_encoder(device=a.device)
        print("REAL encoders: Wav2Vec 2.0 base (audio) + ResNet-18 ImageNet (video, PLACEHOLDER for the lip encoder, D-02).")
    else:
        from features import standins
        venc, aenc = standins.video_encoder, standins.audio_encoder
        print("STAND-IN encoders: for testing only.")
    clips = pd.read_csv(a.clips)
    n = build(clips, a.cache, venc, aenc)
    write_manifest(clips, a.out_manifest)
    print(f"wrote {n} feature files to {a.cache} and manifest {a.out_manifest}")
    print(clips.groupby(["split", "y"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()