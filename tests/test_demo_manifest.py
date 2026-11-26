import pytest
from data.make_demo_manifest import build


def make_tree(root, pairs=("pair1", "pair2", "pair3"), drop=None):
    for p in pairs:
        d = root / p; d.mkdir(parents=True)
        for n in ["source", "target", "fake_320", "fake_256", "fake_160a", "fake_160b"]:
            if (p, n) != drop:
                (d / f"{n}.mp4").write_bytes(b"x")


def test_labels_splits_and_counts(tmp_path):
    make_tree(tmp_path)
    df = build(tmp_path, {"pair1": "train", "pair2": "val", "pair3": "test"})
    assert (df.label == 0).sum() == 6 and (df.label == 1).sum() == 12
    assert df.groupby("identity_id").split.nunique().max() == 1
    assert set(df[df.label == 0].category) == {"real"}


def test_unknown_name_and_missing_folder_and_empty_class(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "pair1" / "mystery.mp4").write_bytes(b"x")
    with pytest.raises(ValueError):
        build(tmp_path, {"pair1": "train", "pair2": "val", "pair3": "test"})
    (tmp_path / "pair1" / "mystery.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        build(tmp_path, {"pair1": "train", "pair2": "val", "pair9": "test"})


def test_split_without_real_video_is_refused(tmp_path):
    make_tree(tmp_path)
    for n in ("source", "target"):
        (tmp_path / "pair3" / f"{n}.mp4").unlink()
    with pytest.raises(ValueError):
        build(tmp_path, {"pair1": "train", "pair2": "val", "pair3": "test"})