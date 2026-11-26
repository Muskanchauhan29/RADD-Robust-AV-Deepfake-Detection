import pandas as pd
import pytest

pytest.importorskip("streamlit")
import app as A


def test_list_runs_needs_best_and_threshold(tmp_path):
    for v, files in [("C", ["best.pt", "threshold.json"]), ("D0", ["best.pt"]), ("D1", ["threshold.json"])]:
        d = tmp_path / v / "seed0"; d.mkdir(parents=True)
        for f in files:
            (d / f).write_text("x")
    assert [r.parent.name for r in A.list_runs(tmp_path)] == ["C"]
    assert A.list_runs(tmp_path / "nothing") == []


def test_demo_videos_labels_and_truth(tmp_path):
    rows = [dict(video_id="a", identity_id="p", label=0, category="real", path="demo_videos/pair1/source.mp4", split="test"),
            dict(video_id="b", identity_id="p", label=1, category="f", path="demo_videos/pair1/fake_1.mp4", split="train")]
    pd.DataFrame(rows).to_csv(tmp_path / "videos_split.csv", index=False)
    d = A.demo_videos(tmp_path / "videos_split.csv", tmp_path)
    assert len(d) == 2 and {v[2] for v in d.values()} == {"real", "fake"} and {v[1] for v in d.values()} == {"train", "test"}
    assert A.demo_videos(tmp_path / "missing.csv", tmp_path) == {}