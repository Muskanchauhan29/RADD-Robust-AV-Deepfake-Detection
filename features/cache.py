"""Step 3: cache layout [Proposed] features/cache/{split}/{view}/{clip_id}.pt

Each file holds {"video": [T, Dv], "audio": [T, Da]} in half precision, T = 50 (2 s at 25 fps).
"""
from pathlib import Path
import torch

T = 50
VIEWS = ("clean", "degraded", "desync", "heldout_audio", "heldout_video", "heldout_joint")   # last three: test-time conditions (D-14)
SPLITS = ("train", "val", "test")


def cache_path(root, split: str, view: str, clip_id: str) -> Path:
    if split not in SPLITS or view not in VIEWS:
        raise ValueError(f"bad split/view: {split}/{view}")
    return Path(root) / split / view / f"{clip_id}.pt"


def save_features(path, video: torch.Tensor, audio: torch.Tensor) -> None:
    _check(video, audio)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"video": video.half().cpu(), "audio": audio.half().cpu()}, path)


def load_features(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["video"], d["audio"]


def _check(video: torch.Tensor, audio: torch.Tensor) -> None:
    for name, x in (("video", video), ("audio", audio)):
        if x.ndim != 2 or x.shape[0] != T:
            raise ValueError(f"{name} features must be [{T}, D], got {tuple(x.shape)}")
        if not torch.isfinite(x.float()).all():
            raise ValueError(f"{name} features contain NaN or inf")


def validate_file(path) -> None:
    """Acceptance check of step 3: 50 video and 50 audio frames, half precision, finite."""
    v, a = load_features(path)
    _check(v, a)
    if v.dtype != torch.float16 or a.dtype != torch.float16:
        raise ValueError("features must be stored in half precision")
