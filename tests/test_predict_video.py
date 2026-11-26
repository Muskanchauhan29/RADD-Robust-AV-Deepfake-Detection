import subprocess
import pandas as pd
import pytest
import build_features as B
import predict_video as PV
import run_all as RA
from features import standins
from views.codecs import have_ffmpeg
import numpy as np


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pv"); rng = np.random.default_rng(0); rows = []
    for split, nr, nf in [("train", 7, 12), ("val", 6, 16), ("test", 6, 8)]:
        for y, n in [(0, nr), (1, nf)]:
            for i in range(n):
                cid = f"{split}_{y}_{i}_c0"; p = tmp / f"{cid}.npz"
                np.savez(p, frames=rng.integers(0, 255, (50, 48, 48, 3), dtype=np.uint8), wav=(0.1 * rng.standard_normal(32000)).astype(np.float32))
                rows.append(dict(clip_id=cid, video_id=cid[:-3], identity_id=split, y=y, category="x", split=split, path=str(p)))
    clips = pd.DataFrame(rows)
    B.build(clips, tmp / "cache", standins.video_encoder, standins.audio_encoder)
    return B.write_manifest(clips, tmp / "m" / "clips.csv"), tmp / "cache", tmp


@pytest.fixture(scope="module")
def run_dir(world):
    m, cache, tmp = world
    return RA.run_one("D1", 0, m, cache, tmp / "res_pv", 200, batch_size=8, max_epochs=1)


@pytest.fixture
def video(tmp_path):
    if not have_ffmpeg():
        pytest.skip("ffmpeg missing")
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=4.5",
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100:duration=4.5", "-shortest", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_predict_video_returns_scores_threshold_and_verdict(run_dir, video):
    r = PV.predict_video(video, run_dir, standins.video_encoder, standins.audio_encoder, size=48)
    assert len(r["clips"]) == 2 and r["clips"].p_fake.between(0, 1).all()                 # 4.5 s -> two 2 s clips
    assert r["verdict"] in ("FAKE", "REAL") and (r["verdict"] == "FAKE") == (r["mean_p_fake"] >= r["threshold"])
    import json
    assert r["threshold"] == json.load(open(run_dir / "threshold.json"))["threshold"]       # read, not recomputed


def test_short_video_is_refused(run_dir, tmp_path):
    if not have_ffmpeg():
        pytest.skip("ffmpeg missing")
    p = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100:duration=1", "-shortest", "-pix_fmt", "yuv420p", str(p)], check=True)
    with pytest.raises(ValueError):
        PV.predict_video(p, run_dir, standins.video_encoder, standins.audio_encoder, size=48)


def test_predict_all_table(run_dir, video, tmp_path):
    import predict_all_demo as PA
    m = pd.DataFrame([dict(label=0, path=str(video), split="test"), dict(label=1, path=str(video), split="train")])
    df = PA.predict_all(m, run_dir, standins.video_encoder, standins.audio_encoder, size=48)
    assert list(df.columns) == ["video", "split", "truth", "verdict", "correct", "mean_p_fake", "threshold", "n_clips"]
    assert set(df.truth) == {"real", "fake"} and df.correct.isin([0, 1]).all() and (df.n_clips == 2).all()
