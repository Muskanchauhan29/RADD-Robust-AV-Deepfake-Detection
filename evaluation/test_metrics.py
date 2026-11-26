import numpy as np
import pandas as pd
import pytest
from evaluation import metrics as M
from evaluation import stats as S


def frame(y, score, video=None, thr=0.5, cat="a"):
    y = np.asarray(y); n = len(y)
    df = pd.DataFrame({"clip_id": range(n), "video_id": video if video is not None else range(n),
                       "y": y, "score": score, "category": cat})
    return M.add_pred(df, thr)


# ------------------------------------------------------------------ threshold and metrics
def test_perfect_inverted_and_random_scores():
    y = np.array([0] * 50 + [1] * 50)
    perfect = np.r_[np.full(50, 0.1), np.full(50, 0.9)]
    t, eer = M.eer_threshold(y, perfect)
    assert eer == 0 and 0.1 < t <= 0.9
    assert M.evaluate_condition(frame(y, perfect))["auc"] == 1
    assert M.evaluate_condition(frame(y, 1 - perfect))["auc"] == 0
    rnd = np.random.default_rng(0).random(2000)
    yr = np.random.default_rng(1).integers(0, 2, 2000)
    assert abs(M.evaluate_condition(frame(yr, rnd))["auc"] - 0.5) < 0.05
    assert abs(M.eer_threshold(yr, rnd)[1] - 0.5) < 0.05


def test_eer_point_is_balanced_on_gaussians():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(5000), np.ones(5000)]
    s = np.r_[rng.normal(0.3, 0.1, 5000), rng.normal(0.7, 0.1, 5000)]
    t, eer = M.eer_threshold(y, s)
    d = M.add_pred(pd.DataFrame({"clip_id": range(10000), "video_id": range(10000), "y": y, "score": s}), t)
    assert abs(t - 0.5) < 0.02 and abs(M.dr_fpr(d) - (1 - M.sensitivity(d))) < 0.02
    with pytest.raises(ValueError):
        M.eer_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_hand_computed_example():
    #        real                fake
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.6, 0.7, 0.2, 0.9, 0.8, 0.4, 0.55]
    df = frame(y, s, thr=0.5)
    assert M.dr_fpr(df) == 0.5 and M.sensitivity(df) == 0.75
    assert M.confusion(df) == {"tn": 2, "fp": 2, "fn": 1, "tp": 3}
    r = M.evaluate_condition(df)
    assert r["accuracy"] == 5 / 8 and abs(r["f1"] - 2 * 3 / (2 * 3 + 2 + 1)) < 1e-9
    assert sum(M.confusion(df).values()) == len(df)


def test_dr_fpr_uses_real_clips_only_and_empty_is_nan():
    assert np.isnan(M.dr_fpr(frame([1, 1], [0.9, 0.8])))
    assert np.isnan(M.sensitivity(frame([0, 0], [0.1, 0.2])))


def test_drift_distance_and_per_category():
    assert M.paired_drift(np.zeros((3, 5)), np.full((3, 5), 0.2)) == pytest.approx(0.2)
    e = np.zeros((2, 4, 3)); e[..., 0] = 1
    assert M.embedding_cosine_distance(e, e) == pytest.approx(0)
    o = np.zeros_like(e); o[..., 1] = 1
    assert M.embedding_cosine_distance(e, o) == pytest.approx(1)
    df = frame([0, 1, 0, 1], [0.1, 0.9, 0.9, 0.1], cat=["a", "a", "b", "b"])
    pc = M.per_category(df)
    assert pc.loc["a", "accuracy"] == 1 and pc.loc["b", "accuracy"] == 0


# ------------------------------------------------------------------ bootstrap and decision rule
def real_frames(n_videos, bad_a, bad_b, shift_b=0):
    """2 real clips per video; the first `bad` videos are false alarms (pred = 1).
    shift_b moves the false alarms of b to other videos so the two sets are not nested."""
    def mk(bad, shift=0):
        pred = np.repeat(np.roll((np.arange(n_videos) < bad).astype(int), shift), 2)
        return pd.DataFrame({"clip_id": range(2 * n_videos), "video_id": np.repeat(np.arange(n_videos), 2),
                             "y": 0, "score": pred.astype(float), "pred": pred})
    return mk(bad_a), mk(bad_b, shift_b)


def test_identical_frames_give_zero_width_interval():
    a, b = real_frames(40, 10, 10)
    assert S.bootstrap_delta(a, b, M.dr_fpr) == (0.0, 0.0)


def test_clear_improvement_has_negative_interval():
    a, b = real_frames(100, 10, 40)                     # FPR 0.10 vs 0.40
    ci = S.bootstrap_delta(a, b, M.dr_fpr, seed=1)
    assert ci[1] < 0 and S.classify(ci, delta=0.02) == "improved"


def test_swapping_direction_gives_worse():
    a, b = real_frames(100, 10, 40)
    ci = S.bootstrap_delta(b, a, M.dr_fpr, seed=1)
    assert ci[0] > 0 and S.classify(ci, 0.02) == "worse"


def test_small_noisy_difference_is_inconclusive_not_equivalent():
    a, b = real_frames(10, 2, 3, shift_b=5)             # 10 real videos only: wide interval containing 0
    ci = S.bootstrap_delta(a, b, M.dr_fpr, seed=2)
    assert ci[0] < 0 < ci[1]
    assert not S.improved(ci) and not S.worse(ci)
    assert S.classify(ci, delta=0.02) == "inconclusive"


def test_bootstrap_is_deterministic_and_checks_pairing():
    a, b = real_frames(30, 5, 9)
    assert S.bootstrap_delta(a, b, M.dr_fpr, seed=3) == S.bootstrap_delta(a, b, M.dr_fpr, seed=3)
    with pytest.raises(ValueError):
        S.bootstrap_delta(a[a.video_id < 20], b, M.dr_fpr)


def test_nan_resamples_give_undefined_interval_which_is_inconclusive():
    a = pd.DataFrame({"clip_id": range(10), "video_id": range(10), "y": [0] + [1] * 9,
                      "score": 0.0, "pred": 0})
    ci = S.bootstrap_delta(a, a.copy(), M.dr_fpr, seed=0)       # one real video: most resamples have none
    assert np.isnan(ci[0]) and S.classify(ci, 0.05) == "inconclusive"


def test_decision_helpers_follow_the_corrected_rule():
    assert S.classify((-0.10, -0.02), 0.05) == "improved"
    assert S.classify((0.01, 0.10), 0.05) == "worse"
    assert S.classify((-0.01, 0.02), 0.05) == "equivalent"       # inside (-0.05, 0.05)
    assert S.classify((-0.08, 0.07), 0.05) == "inconclusive"     # contains zero but wider than the margin
    assert not S.equivalent((-0.08, 0.07), 0.05)
    assert S.sens_ok((-0.005, 0.02), 0.01) and not S.sens_ok((-0.02, 0.02), 0.01)
