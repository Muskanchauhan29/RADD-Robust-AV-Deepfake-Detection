import pytest
from data.make_synthetic_manifest import make
from data.split_fakeavceleb import split_by_identity, split_counts, check_split

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_no_identity_in_two_splits():
    df = split_by_identity(make(100), RATIOS, seed=0)
    assert df.groupby("identity_id")["split"].nunique().max() == 1
    check_split(df, min_real=5)


def test_counts_printed_and_cover_all_videos(capsys):
    raw = make(100)
    df = split_by_identity(raw, RATIOS, seed=0)
    t = split_counts(df)
    print(t.to_string())
    assert t[["real", "fake"]].to_numpy().sum() == len(raw)
    assert "real" in capsys.readouterr().out


def test_ratios_roughly_respected():
    df = split_by_identity(make(200), RATIOS, seed=0)
    share = df["split"].value_counts(normalize=True)
    for s, r in RATIOS.items():
        assert abs(share[s] - r) < 0.05


def test_deterministic_for_same_seed_and_different_for_other():
    a = split_by_identity(make(100), RATIOS, seed=1)
    b = split_by_identity(make(100), RATIOS, seed=1)
    c = split_by_identity(make(100), RATIOS, seed=2)
    assert a["split"].equals(b["split"])
    assert not a["split"].equals(c["split"])


def test_stops_when_a_split_has_too_few_real_videos():
    df = split_by_identity(make(6), RATIOS, seed=0)      # tiny dataset
    with pytest.raises(AssertionError):
        check_split(df, min_real=50)


def test_leak_is_detected():
    df = split_by_identity(make(100), RATIOS, seed=0)
    df.loc[df.index[0], "split"] = "test" if df.loc[df.index[0], "split"] != "test" else "val"
    with pytest.raises(AssertionError):
        check_split(df, min_real=1)
