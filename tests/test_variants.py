import pytest
import torch
import train_step as TS
from config import load_config, load_dict
from model.heads import RACL

# Table 10 of the Guide: (degraded_cls, nce_deg_pos, pair_loss)
TABLE_10 = {"C": (False, False, False), "D0": (True, False, False), "D1a": (True, True, False),
            "D1b": (True, False, True), "D1": (True, True, True)}
B, T, DV, DA = 6, 50, 16, 24


def flags(name):
    f = load_config(name).flags
    return (f.degraded_cls, f.nce_deg_pos, f.pair_loss)


@pytest.mark.parametrize("name", TABLE_10)
def test_config_flags_match_table_10(name):
    assert flags(name) == TABLE_10[name]


def test_variants_differ_only_in_flags_and_there_are_five():
    assert len(TABLE_10) == 5
    base = {k: v for k, v in load_dict("C").items() if k != "flags"}
    for name in TABLE_10:
        assert {k: v for k, v in load_dict(name).items() if k != "flags"} == base
    assert len({flags(n) for n in TABLE_10}) == 5                  # all distinct


def test_base_has_all_flags_off():
    f = load_dict("base")["flags"]
    assert not any(f.values())


class Batch(dict):
    """Batch that records which keys were read; degraded keys can be forbidden."""
    def __init__(self, forbid=(), **kw):
        super().__init__(**kw); self.forbid, self.read = set(forbid), set()
    def __getitem__(self, k):
        assert k not in self.forbid, f"variant touched forbidden data: {k}"
        self.read.add(k); return super().__getitem__(k)


def make_batch(forbid=()):
    g = torch.Generator().manual_seed(0)
    r = lambda d: torch.randn(B, T, d, generator=g)
    return Batch(forbid, y=torch.tensor([0, 1, 0, 1, 0, 1]),
                 v_clean=r(DV), a_clean=r(DA), v_deg=r(DV), a_deg=r(DA), v_desync=r(DV), a_desync=r(DA))


class Spy:
    def __init__(self, monkeypatch, model):
        self.calls = {"model": 0, "nce_np": None, "pair": 0, "cls_n": None}
        orig_fwd, orig_nce, orig_pair, orig_cls = model.forward, TS.nce_loss, TS.pair_loss, TS.cls_loss
        def fwd(*a, **k): self.calls["model"] += 1; return orig_fwd(*a, **k)
        def nce(sp, sn, tau): self.calls["nce_np"] = sp.shape[1]; return orig_nce(sp, sn, tau)
        def pair(*a): self.calls["pair"] += 1; return orig_pair(*a)
        def cls(z, y): self.calls["cls_n"] = len(z); return orig_cls(z, y)
        monkeypatch.setattr(model, "forward", fwd)
        monkeypatch.setattr(TS, "nce_loss", nce); monkeypatch.setattr(TS, "pair_loss", pair)
        monkeypatch.setattr(TS, "cls_loss", cls)


def run(name, monkeypatch, forbid=()):
    torch.manual_seed(0)
    m = RACL(DV, DA, d=32, layers=1)
    spy = Spy(monkeypatch, m)
    loss, parts = TS.training_step(m, make_batch(forbid), load_config(name))
    loss.backward()
    return m, spy.calls, loss, parts


def test_C_never_loads_degraded_data(monkeypatch):
    _, calls, loss, parts = run("C", monkeypatch, forbid=("v_deg", "a_deg"))   # raises if touched
    assert calls["model"] == 2 and calls["cls_n"] == B and calls["nce_np"] == 1
    assert calls["pair"] == 0 and parts["pair"] is None and torch.isfinite(loss)


@pytest.mark.parametrize("name,exp", [
    # name: (model forward passes, L_cls batch, NCE positives per frame, L_pair calls)
    ("C", (2, B, 1, 0)), ("D0", (3, 2 * B, 1, 0)), ("D1a", (3, 2 * B, 2, 0)),
    ("D1b", (3, 2 * B, 1, 1)), ("D1", (3, 2 * B, 2, 1))])
def test_each_variant_runs_exactly_its_terms(name, exp, monkeypatch):
    _, c, loss, parts = run(name, monkeypatch)
    assert (c["model"], c["cls_n"], c["nce_np"], c["pair"]) == exp
    assert torch.isfinite(loss) and parts["cls"] is not None and parts["nce"] is not None
    assert (parts["pair"] is not None) == (TABLE_10[name][2])


def test_desync_clips_are_never_in_L_cls_and_only_real_enter_nce(monkeypatch):
    # Poison fake-clip rows of the desync tensors: the loss must not change, because only real rows are used.
    torch.manual_seed(0); m = RACL(DV, DA, d=32, layers=1); cfg = load_config("D1")
    b1 = make_batch(); b2 = make_batch()
    fake = b2["y"] == 1
    b2["v_desync"][fake] = 1e3; b2["a_desync"][fake] = -1e3
    m.eval()
    l1, _ = TS.training_step(m, b1, cfg); l2, _ = TS.training_step(m, b2, cfg)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_pair_loss_does_not_train_through_the_clean_branch(monkeypatch):
    # With only L_pair active (lambda1 = 0, cls ignored via D1b weights) the clean-view score gets no gradient
    # from the pair term; checked directly on the term.
    torch.manual_seed(0); m = RACL(DV, DA, d=32, layers=1)
    b = make_batch(); oc, od = m(b["v_clean"], b["a_clean"]), m(b["v_deg"], b["a_deg"])
    oc["c"].retain_grad()
    TS.pair_loss(oc["c"], od["c"], b["y"] == 0).backward()
    assert oc["c"].grad is None or oc["c"].grad.abs().sum() == 0


def test_batch_without_real_clips_does_not_crash(monkeypatch):
    torch.manual_seed(0); m = RACL(DV, DA, d=32, layers=1)
    b = make_batch(); b["y"] = torch.ones(B, dtype=torch.long)
    loss, parts = TS.training_step(m, b, load_config("D1"))
    assert torch.isfinite(loss) and parts["nce"] is None and parts["pair"] == 0.0
