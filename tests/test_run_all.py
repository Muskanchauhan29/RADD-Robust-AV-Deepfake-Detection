import numpy as np
import pandas as pd
import pytest
import build_features as B
import run_all as RA
from features import standins


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ra"); rng = np.random.default_rng(0); rows = []
    for split, nr, nf in [("train", 7, 12), ("val", 6, 16), ("test", 6, 8)]:       # same counts as the real pilot
        for y, n in [(0, nr), (1, nf)]:
            for i in range(n):
                cid = f"{split}_{y}_{i}_c0"; p = tmp / f"{cid}.npz"
                np.savez(p, frames=rng.integers(0, 255, (50, 48, 48, 3), dtype=np.uint8), wav=(0.1 * rng.standard_normal(32000)).astype(np.float32))
                rows.append(dict(clip_id=cid, video_id=cid[:-3], identity_id=split, y=y, category="real" if y == 0 else "fake", split=split, path=str(p)))
    clips = pd.DataFrame(rows)
    B.build(clips, tmp / "cache", standins.video_encoder, standins.audio_encoder)
    return B.write_manifest(clips, tmp / "m" / "clips.csv"), tmp / "cache", tmp


def test_run_one_writes_files_and_is_resumable(world, capsys):
    m, cache, tmp = world
    run = RA.run_one("D1", 0, m, cache, tmp / "res", 200, batch_size=8, max_epochs=1)
    for f in ["config.yaml", "train.log", "training_history.csv", "best.pt", "threshold.json", "metrics.csv"]:
        assert (run / f).exists(), f
    RA.run_one("D1", 0, m, cache, tmp / "res", 200, batch_size=8, max_epochs=1)
    assert "skip D1 seed 0" in capsys.readouterr().out


def test_every_variant_gets_the_same_overrides(world):
    import yaml
    m, cache, tmp = world
    cfgs = []
    for v in ["C", "D0"]:
        run = RA.run_one(v, 0, m, cache, tmp / "res2", 160, batch_size=8, max_epochs=1)
        cfgs.append(yaml.safe_load(open(run / "config.yaml"))["config"])
    assert cfgs[0]["desync"] == cfgs[1]["desync"] == {"max_shift_ms": 160}
    assert cfgs[0]["train"] == cfgs[1]["train"] and cfgs[0]["flags"] != cfgs[1]["flags"]


def test_shift_is_required():
    import sys
    with pytest.raises(SystemExit):
        sys.argv = ["run_all.py"]; RA.main()