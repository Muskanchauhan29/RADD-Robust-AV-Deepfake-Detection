import math
import torch
import torch.nn.functional as F
import pytest
from losses import cls_loss, nce_loss, pair_loss

TAU = 0.07


# ---------------------------------------------------------------- L_cls
def test_cls_matches_manual_bce():
    z, y = torch.tensor([2.0, -1.0, 0.3]), torch.tensor([1, 0, 1])
    manual = -(y * torch.log(torch.sigmoid(z)) + (1 - y) * torch.log(1 - torch.sigmoid(z))).mean()
    assert torch.allclose(cls_loss(z, y), manual, atol=1e-6)


def test_cls_prefers_correct_side_and_checks_shapes():
    y = torch.tensor([1, 0])
    assert cls_loss(torch.tensor([3.0, -3.0]), y) < cls_loss(torch.tensor([-3.0, 3.0]), y)
    with pytest.raises(ValueError):
        cls_loss(torch.zeros(3), y)


# ---------------------------------------------------------------- L_NCE
def test_nce_aligned_lower_than_misaligned():
    aligned = nce_loss(torch.full((8, 1), 0.9), torch.full((8, 1), 0.1))
    swapped = nce_loss(torch.full((8, 1), 0.1), torch.full((8, 1), 0.9))
    assert aligned < swapped and aligned < 0.01


def test_nce_equals_softplus_for_one_pos_one_neg():
    sp, sn = torch.tensor([[0.6]]), torch.tensor([[0.2]])
    expect = F.softplus((sn - sp) / TAU)[0, 0]
    assert torch.allclose(nce_loss(sp, sn), expect, atol=1e-6)


def test_nce_matches_formula_with_two_positives():
    sp, sn = torch.tensor([[0.6, 0.5]]), torch.tensor([[0.2]])
    num = (sp / TAU).exp().sum(); den = num + (sn / TAU).exp().sum()
    assert torch.allclose(nce_loss(sp, sn), -torch.log(num / den), atol=1e-5)


def test_second_positive_lowers_the_loss():
    sn = torch.tensor([[0.3]])
    one = nce_loss(torch.tensor([[0.5]]), sn)
    two = nce_loss(torch.tensor([[0.5, 0.5]]), sn)
    assert two < one


def test_nce_gradients_push_the_right_way_and_empty_batch_is_zero():
    sp = torch.tensor([[0.5]], requires_grad=True); sn = torch.tensor([[0.3]], requires_grad=True)
    nce_loss(sp, sn).backward()
    assert sp.grad < 0 and sn.grad > 0                  # raise positive sim, lower negative sim
    assert nce_loss(torch.zeros(0, 1), torch.zeros(0, 1)) == 0
    with pytest.raises(ValueError):
        nce_loss(torch.zeros(3), torch.zeros(3, 1))


# ---------------------------------------------------------------- L_pair
def test_pair_zero_when_scores_identical():
    c = torch.rand(4, 50)
    assert pair_loss(c, c.clone(), torch.ones(4, dtype=torch.bool)) == 0


def test_pair_value_and_real_only_mask():
    cc = torch.full((2, 10), 0.8)
    cd = torch.stack([torch.full((10,), 0.5), torch.full((10,), 0.0)])   # clip 0 gap 0.3, clip 1 gap 0.8
    real = torch.tensor([True, False])
    assert torch.allclose(pair_loss(cc, cd, real), torch.tensor(0.3), atol=1e-6)   # fake clip ignored
    assert pair_loss(cc, cd, torch.tensor([False, False])) == 0                    # no real clips: 0, not NaN


def test_pair_clean_branch_is_stop_gradient():
    cc = torch.rand(3, 20, requires_grad=True); cd = torch.rand(3, 20, requires_grad=True)
    pair_loss(cc, cd, torch.ones(3, dtype=torch.bool)).backward()
    assert cc.grad is None or cc.grad.abs().sum() == 0
    assert cd.grad.abs().sum() > 0


# ---------------------------------------------------------------- with the model
def test_losses_reach_the_model_parameters():
    from model.heads import RACL
    torch.manual_seed(0)
    m = RACL(16, 24, d=32, layers=1)
    fv, fa = torch.randn(4, 50, 16), torch.randn(4, 50, 24)
    oc, od, os_ = m(fv, fa), m(fv + 0.1, fa), m(fv, fa.roll(5, 1))
    s = lambda o: (o["ev"] * o["ea"]).sum(-1).flatten().unsqueeze(-1)
    nce = nce_loss(torch.cat([s(oc), s(od)], 1), s(os_))
    pair = pair_loss(oc["c"], od["c"], torch.tensor([True, True, False, False]))
    m.zero_grad(); nce.backward(retain_graph=True)
    assert m.proj_v.weight.grad.abs().sum() > 0 and m.pv.weight.grad.abs().sum() > 0
    m.zero_grad(); pair.backward()
    assert m.score.weight.grad.abs().sum() > 0
    assert m.proj_v.weight.grad is None or m.proj_v.weight.grad.abs().sum() == 0   # L_pair never touches embeddings
