import json
import numpy as np
import pandas as pd
import pytest
import pilot_synthetic as P
import make_report as R
from evaluation import stats as S

VARS, SEEDS = ["C", "D0", "D1a", "D1b", "D1"], (0, 1)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rep")
    clips, cache = P.build_world(tmp, n_identities=30)
    for v in VARS:
        for s in SEEDS:
            P.run_variant(v, clips, cache, tmp, epochs=1, seed=s)
    return tmp / "results"


def test_report_files_and_unset_margins(world, tmp_path):
    ci, verdict = R.make_report(world, tmp_path, VARS, SEEDS, n_boot=100)
    for f in ["metrics_all.csv", "metrics_mean_std.csv", "bootstrap_ci.csv", "table_h1.csv", "table_h2.csv", "h2_verdict.json",
              "training_curves.png", "comparison_drfpr.png", "comparison_sens_vs_drfpr.png", "comparison_drift.png",
              "comparison_auc_eer.png", "roc_overlay_clean.png", "roc_overlay_heldout_joint.png"]:
        assert (tmp_path / f).exists(), f
    assert verdict["outcome"].startswith("undetermined")
    assert set(ci.label) <= {R.UNSET, "inconclusive (undefined interval)"}
    assert {"decision", "descriptive", "h1"} <= set(ci.kind)
    assert json.load(open(tmp_path / "h2_verdict.json"))["outcome"] == verdict["outcome"]


def test_with_margins_labels_and_verdict(world, tmp_path):
    ci, verdict = R.make_report(world, tmp_path, VARS, SEEDS, n_boot=100, delta=0.05, sens_margin=0.05, h1_margin=0.02)
    assert not (ci.label == R.UNSET).any()
    assert verdict["outcome"] in {"supported", "redundant (D1a equivalent to D1)", "not supported", "inconclusive"}


def test_missing_run_raises(world, tmp_path):
    with pytest.raises(FileNotFoundError):
        R.make_report(world, tmp_path, VARS, (0, 1, 9), n_boot=10)


def test_non_identical_test_clips_detected(world):
    _, preds = R.load_runs(world, VARS, SEEDS)
    k = ("D1", 0, "clean"); preds[k] = preds[k].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        R.check_identical_test_clips(preds)


def test_vectorised_bootstrap_agrees_with_reference(world):
    _, preds = R.load_runs(world, VARS, SEEDS)
    a, b = preds[("D1b", 0, "clean")], preds[("D0", 0, "clean")]
    fpr = lambda d: float(d[d.y == 0].prediction.mean())
    ref = S.bootstrap_delta(a, b, fpr, n_boot=400)
    ci = R.bootstrap([a], [b], "dr_fpr", n_boot=400)
    pt = R.point([a], "dr_fpr") - R.point([b], "dr_fpr")
    assert pt == pytest.approx(fpr(a) - fpr(b))
    # different random streams, so compare loosely: same centre, similar width
    assert abs(np.mean(ci) - np.mean(ref)) < 0.1 and abs((ci[1] - ci[0]) - (ref[1] - ref[0])) < 0.1


def test_predictions_have_drift_columns(world):
    d = pd.read_csv(world / "C" / "seed0" / "predictions_heldout_audio.csv")
    assert d.drift.notna().all() and (d.emb_dist.notna()).all()
    assert pd.read_csv(world / "C" / "seed0" / "predictions_clean.csv").drift.isna().all()
