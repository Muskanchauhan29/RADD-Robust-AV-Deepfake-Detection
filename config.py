"""Load a variant config: `defaults: base` is merged under the variant's own keys. Returns nested namespaces."""
from pathlib import Path
from types import SimpleNamespace
import yaml

CONFIG_DIR = Path(__file__).parent / "configs"


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_dict(name_or_path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(open(p))
    base = cfg.pop("defaults", None)
    return _merge(load_dict(p.parent / f"{base}.yaml"), cfg) if base else cfg


def to_ns(d):
    return SimpleNamespace(**{k: to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(name_or_path):
    return to_ns(load_dict(name_or_path))
