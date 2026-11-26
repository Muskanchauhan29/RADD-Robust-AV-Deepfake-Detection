"""Shared inference helpers for threshold.py and evaluate.py (no training, no threshold logic here)."""
from pathlib import Path
import torch
from features.cache import cache_path, load_features
from model.heads import RACL


def load_run_model(run_dir, device="cpu"):
    ck = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    m, d = ck["config"]["model"], ck["dims"]
    model = RACL(d["dv"], d["da"], d=m["d"], layers=m["layers"], heads=m["heads"], w=m["window"],
                 bias_range=m["bias_range"], proj_dim=m["proj_dim"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


def load_view(cache_dir, split, view, clip_ids):
    """Stack cached features of the given clips for one view -> ([N,T,Dv], [N,T,Da]) float32.
    Raises FileNotFoundError if any file is missing (a missing condition must never be skipped silently)."""
    vs, as_ = [], []
    for c in clip_ids:
        v, a = load_features(cache_path(cache_dir, split, view, c))
        vs.append(v.float()); as_.append(a.float())
    return torch.stack(vs), torch.stack(as_)


@torch.no_grad()
def predict(model, v, a, batch_size=64, device="cpu"):
    out = {"logit": [], "c": [], "ev": [], "ea": []}
    for i in range(0, len(v), batch_size):
        o = model(v[i:i + batch_size].to(device), a[i:i + batch_size].to(device))
        for k in out: out[k].append(o[k].cpu())
    return {k: torch.cat(x) for k, x in out.items()}
