import json
import numpy as np
import pandas as pd
import pytest
import torch
import evaluate as EV
import threshold as TH
import train as TR
from config import load_config
from features import cache

T, DV, DA = 50, 12, 10
CONDS = ["heldout_audio", "heldout_video", "heldout_joint"]


def build(root, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for split, nr, nf in (("train", 10, 30), ("val", 8, 16), ("test", 8, 16)):
        for i in range(nr + nf):
            y = 0 if i < nr else 1
            rows.append(dict(clip_id=f"{split}{i}", video_id=f"v{split}{i}", identity_id=f"i{split}{i}", y=y,
                             category="real" if y == 0 else "fakeA", split=split, degradation="noise"))
    df = pd.DataFrame(rows)
    for r in df.itertuples():
        base = rng.standard_normal((T, DV)) + 1.5 * r.y; au = rng.standard_normal((T, DA))
        views = {"clean": 0, "degraded": 0.3, **{c: 0.3 for c in CONDS}}
        for view, noise in views.items():
            if view not in ("clean", "degraded") and r.split != "test":
                continue
            v = base + noise * rng.standard_normal((T, DV))
            cache.save_features(cache.cache_path(root, r.split, view, r.clip_id), torch.from_numpy(v), torch.from_numpy(au))
    return df


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache"); df = build(root)
    cfg = load_config("D1"); cfg.data.cache_dir = str(root)
    cfg.model.d, cfg.model.layers, cfg.model.proj_dim = 32, 1, 16
    cfg.train.batch_size, cfg.train.max_epochs, cfg.train.lr, cfg.desync.max_shift_ms = 8, 3, 1e-3, 200
    run = TR.run_training(cfg, df, tmp_path_factory.mktemp("run"), 0, "cpu")
    return df, root, run


def fresh_copy(trained, tmp_path):
    import shutil
    df, root, run = trained; dst = tmp_path / "run"; shutil.copytree(run, dst)
    for f in dst.glob("threshold.json"): f.unlink()
    return df, root, dst


def test_threshold_is_saved_once_from_clean_validation_only(trained, tmp_path):
    df, root, run = fresh_copy(trained, tmp_path)
    res = TH.write_threshold(run, df, root)
    saved = json.load(open(run / "threshold.json"))
    assert saved["split"] == "val" and saved["view"] == "clean" and saved["n_real"] == 8 and saved["n_fake"] == 16
    assert 0 <= saved["threshold"] <= 1 and 0 <= saved["val_eer"] <= 0.5 and res == saved
    with pytest.raises(FileExistsError):
        TH.write_threshold(run, df, root)                       # never overwritten


def test_threshold_ignores_test_data(trained, tmp_path):
    import shutil
    df, root, run = fresh_copy(trained, tmp_path)
    a = TH.write_threshold(run, df, root)["threshold"]
    root2 = tmp_path / "cache_copy"; shutil.copytree(root, root2)          # work on a copy: other tests share root
    for f in (root2 / "test").rglob("*.pt"):                                # destroy every test feature file
        torch.save({"video": torch.zeros(T, DV).half(), "audio": torch.zeros(T, DA).half()}, f)
    (run / "threshold.json").unlink()
    assert TH.write_threshold(run, df, root2)["threshold"] == a


def test_evaluate_writes_every_file_and_metrics_columns(trained, tmp_path):
    df, root, run = fresh_copy(trained, tmp_path)
    TH.write_threshold(run, df, root)
    m = EV.evaluate_run(run, df, root)
    assert list(m.condition) == EV.CONDITIONS and set(m.variant) == {"D1"} and set(m.seed) == {0}
    for c in EV.CONDITIONS:
        for f in (f"predictions_{c}.csv", f"roc_{c}.png", f"confusion_{c}.png"):
            assert (run / f).exists(), f
    assert (run / "metrics.csv").exists()
    for col in ("auc", "eer", "accuracy", "f1", "dr_fpr", "sensitivity", "paired_drift", "emb_cos_dist"):
        assert col in m.columns
    p = pd.read_csv(run / "predictions_clean.csv")
    assert list(p.columns[:7]) == ["clip_id", "video_id", "category", "y", "p_fake", "prediction", "mean_c"] and len(p) == 24


def test_predictions_use_the_saved_threshold_and_it_is_never_retuned(trained, tmp_path):
    df, root, run = fresh_copy(trained, tmp_path)
    TH.write_threshold(run, df, root)
    thr = json.load(open(run / "threshold.json"))["threshold"]
    EV.evaluate_run(run, df, root)
    p = pd.read_csv(run / "predictions_heldout_audio.csv")
    assert ((p.p_fake >= thr).astype(int) == p.prediction).all()
    d = json.load(open(run / "threshold.json")); d["threshold"] = 2.0               # impossible threshold
    json.dump(d, open(run / "threshold.json", "w"))
    m = EV.evaluate_run(run, df, root)
    assert (m.dr_fpr == 0).all() and (m.sensitivity == 0).all()                     # nothing predicted fake


def test_drift_is_nan_for_clean_and_positive_for_degraded(trained, tmp_path):
    df, root, run = fresh_copy(trained, tmp_path)
    TH.write_threshold(run, df, root)
    m = EV.evaluate_run(run, df, root).set_index("condition")
    assert np.isnan(m.loc["clean", "paired_drift"])
    assert (m.loc[CONDS, "paired_drift"] > 0).all() and (m.loc[CONDS, "emb_cos_dist"] > 0).all()


def test_identical_degraded_view_gives_zero_drift(trained, tmp_path):
    import shutil
    df, root0, run = fresh_copy(trained, tmp_path)
    root = tmp_path / "cache_copy"; shutil.copytree(root0, root)
    TH.write_threshold(run, df, root)
    for f in (root / "test" / "clean").glob("*.pt"):                # make heldout_audio identical to clean
        shutil.copy(f, root / "test" / "heldout_audio" / f.name)
    m = EV.evaluate_run(run, df, root, conditions=["clean", "heldout_audio"]).set_index("condition")
    assert m.loc["heldout_audio", "paired_drift"] == pytest.approx(0, abs=1e-6)
    assert m.loc["heldout_audio", "emb_cos_dist"] == pytest.approx(0, abs=1e-5)


def test_missing_condition_files_raise_instead_of_skipping(trained, tmp_path):
    import shutil
    df, root, run = fresh_copy(trained, tmp_path)
    TH.write_threshold(run, df, root)
    moved = tmp_path / "moved"; shutil.move(str(root / "test" / "heldout_video"), str(moved))
    try:
        with pytest.raises(FileNotFoundError):
            EV.evaluate_run(run, df, root, conditions=["clean", "heldout_video"])
    finally:
        shutil.move(str(moved), str(root / "test" / "heldout_video"))
