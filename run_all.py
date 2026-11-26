"""Train, threshold, evaluate every variant and seed, then write the report. One command for the pilot (and later the 15 runs).

    python run_all.py --max_shift_ms 200 --batch_size 8 --variants C D0 D1a D1b D1 --seeds 0 --out results

For each (variant, seed): train.run_training -> threshold.write_threshold (once) -> evaluate.evaluate_run -> results/{variant}/seed{s}/.
Then make_report.py is called on the results folder (margins stay unset unless you pass them: D-06).
All variants get EXACTLY the same overrides; they are written into each run's config.yaml.

Pilot overrides (all [Proposed] placeholders, say so in the report):
    --max_shift_ms  desync shift (D-13 open; training refuses without it)
    --batch_size    8 because the pilot has only 12 training fake clips (paper value 32 gives 1 batch per epoch)
Runs that already have metrics.csv are skipped, so an interrupted job can be restarted.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

import evaluate as EV
import make_report as R
import threshold as TH
import train as TR
from config import load_config

ALL = ["C", "D0", "D1a", "D1b", "D1"]


def run_one(variant, seed, manifest, cache_dir, out_root, max_shift_ms, batch_size=None, max_epochs=None, device=None):
    run = Path(out_root) / variant / f"seed{seed}"
    if (run / "metrics.csv").exists():
        print(f"skip {variant} seed {seed}: already done"); return run
    cfg = load_config(variant)
    cfg.data.cache_dir = str(cache_dir); cfg.desync.max_shift_ms = max_shift_ms
    if batch_size is not None:
        cfg.train.batch_size = batch_size
    if max_epochs is not None:
        cfg.train.max_epochs = max_epochs
    t0 = time.time()
    TR.run_training(cfg, manifest, run, seed, device)
    TH.write_threshold(run, manifest, cache_dir)
    EV.evaluate_run(run, manifest, cache_dir)
    print(f"done {variant} seed {seed} in {time.time() - t0:.0f}s")
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/clips.csv"); ap.add_argument("--cache", default="features/cache")
    ap.add_argument("--out", default="results"); ap.add_argument("--variants", nargs="+", default=ALL)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max_shift_ms", type=int, required=True, help="desync shift in ms (D-13: open decision)")
    ap.add_argument("--batch_size", type=int, default=None); ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=None); ap.add_argument("--sens_margin", type=float, default=None)
    ap.add_argument("--h1_margin", type=float, default=None)
    a = ap.parse_args()
    manifest = pd.read_csv(a.manifest)
    print(f"max_shift_ms={a.max_shift_ms} is a PLACEHOLDER for the undecided D-13; batch_size={a.batch_size or 'config'}")
    for s in a.seeds:
        for v in a.variants:
            run_one(v, s, manifest, a.cache, a.out, a.max_shift_ms, a.batch_size, a.max_epochs)
    ci, verdict = R.make_report(a.out, Path(a.out) / "summary", a.variants, tuple(a.seeds), a.n_boot, a.delta, a.sens_margin, a.h1_margin)
    print("H2 outcome:", verdict["outcome"])
    print("report files in", Path(a.out) / "summary")


if __name__ == "__main__":
    main()