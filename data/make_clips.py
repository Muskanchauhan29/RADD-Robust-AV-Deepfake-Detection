"""Step 1 (second half): cut every video of a split manifest into 2 s clips and save the raw clip data.

    python data/make_clips.py --manifest videos_split.csv --out data/clips --crop center

Per clip: 50 frames at 25 fps, [50, S, S, 3] uint8, and 32000 samples of 16 kHz mono audio, saved as clips/{clip_id}.npz.
Writes clips/clips.csv with columns clip_id, video_id, identity_id, y, category, split, start_s, path.
Windows do not overlap; a trailing piece shorter than 2 s is dropped. Every clip inherits its video's split.

The face / mouth crop is undecided (D-02): there is no default. `--crop center` is a PLACEHOLDER (centre square, resized)
that only lets the pipeline run; the report must say so. Pass your own callable via make_clips(..., crop=fn) once D-02 is settled.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from features.cache import T
from views.codecs import SR, have_ffmpeg, read_wav

FPS, CLIP_S = 25, 2
N_SAMP = SR * CLIP_S


def center_crop(frames: np.ndarray, size: int) -> np.ndarray:
    """PLACEHOLDER crop (D-02): centre square, resized to size x size."""
    h, w = frames.shape[1:3]; m = min(h, w); y0, x0 = (h - m) // 2, (w - m) // 2
    return np.stack([cv2.resize(f[y0:y0 + m, x0:x0 + m], (size, size)) for f in frames])


def read_frames(path) -> np.ndarray:
    """All frames as RGB uint8, resampled to 25 fps by nearest-frame selection."""
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or FPS; frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"could not read any frames from {path}")
    frames = np.stack(frames)
    if abs(fps - FPS) > 0.01:
        idx = np.round(np.arange(0, len(frames) / fps, 1 / FPS) * fps).astype(int)
        frames = frames[np.clip(idx[idx < len(frames)], 0, len(frames) - 1)]
    return frames


def read_audio(path) -> np.ndarray:
    """Mono 16 kHz float audio through FFmpeg."""
    if not have_ffmpeg():
        raise RuntimeError("FFmpeg not found on PATH (brew install ffmpeg).")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                            "-c:a", "pcm_s16le", str(out)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr[-300:]}")
        return read_wav(out)


def n_windows(n_frames: int, n_samples: int, max_clips=None) -> int:
    n = min(n_frames // T, n_samples // N_SAMP)
    return n if max_clips is None else min(n, max_clips)


def make_clips(manifest: pd.DataFrame, out_dir, crop, size=96, max_clips_per_video=None, root=".") -> pd.DataFrame:
    """crop: 'center' (placeholder) or a callable frames[T,H,W,3] -> frames[T,S,S,3]. None raises (D-02)."""
    if crop is None:
        raise ValueError("no face/mouth crop chosen (D-02). Use crop='center' only as a stated placeholder.")
    fn = (lambda f: center_crop(f, size)) if crop == "center" else crop
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); rows = []
    for r in manifest.itertuples():
        frames, wav = read_frames(Path(root) / r.path), read_audio(Path(root) / r.path)
        for k in range(n_windows(len(frames), len(wav), max_clips_per_video)):
            f = fn(frames[k * T:(k + 1) * T]); w = wav[k * N_SAMP:(k + 1) * N_SAMP]
            assert f.shape[0] == T and len(w) == N_SAMP
            cid = f"{r.video_id}_c{k}"; path = out_dir / f"{cid}.npz"
            np.savez_compressed(path, frames=f, wav=w.astype(np.float32))
            rows.append(dict(clip_id=cid, video_id=r.video_id, identity_id=r.identity_id, y=int(r.label),
                             category=r.category, split=r.split, start_s=k * CLIP_S, path=str(path)))
    if not rows:
        raise ValueError("no clips produced: every video is shorter than 2 s")
    df = pd.DataFrame(rows); df.to_csv(out_dir / "clips.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="split manifest from split_fakeavceleb.py")
    ap.add_argument("--out", required=True); ap.add_argument("--root", default=".", help="folder the manifest paths are relative to")
    ap.add_argument("--crop", choices=["center"], default=None, help="placeholder crop (D-02); no default")
    ap.add_argument("--size", type=int, default=96); ap.add_argument("--max_clips_per_video", type=int, default=None)
    a = ap.parse_args()
    df = make_clips(pd.read_csv(a.manifest), a.out, a.crop, a.size, a.max_clips_per_video, a.root)
    print(df.groupby(["split", "y"]).size().unstack(fill_value=0).rename(columns={0: "real clips", 1: "fake clips"}).to_string())
    if a.crop == "center":
        print("NOTE: centre-crop is a placeholder for the face/mouth crop (D-02); state this in the report.")


if __name__ == "__main__":
    main()
