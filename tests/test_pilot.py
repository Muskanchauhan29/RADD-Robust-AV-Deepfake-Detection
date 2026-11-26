import pandas as pd
import pytest
import pilot_synthetic as P


@pytest.mark.parametrize("variant", ["C", "D1b"])
def test_pilot_runs_end_to_end_and_writes_every_per_run_file(tmp_path, variant):
    clips, cache_dir = P.build_world(tmp_path, n_identities=30)
    run, metrics = P.run_variant(variant, clips, cache_dir, tmp_path, epochs=2)
    for f in P.PER_RUN_FILES:
        assert (run / f).exists(), f
    assert len(metrics) == 4 and metrics.variant.iloc[0] == variant
    assert len(pd.read_csv(run / "training_history.csv")) >= 1


def test_all_variants_evaluate_the_same_test_clips(tmp_path):
    clips, cache_dir = P.build_world(tmp_path, n_identities=30)
    ids = []
    for v in ["C", "D1"]:
        run, _ = P.run_variant(v, clips, cache_dir, tmp_path, epochs=1)
        ids.append(pd.read_csv(run / "predictions_clean.csv").clip_id.tolist())
    assert ids[0] == ids[1] and len(ids[0]) == (clips.split == "test").sum()
