import torch
import torch.nn.functional as F
from model.windowed_attn import WindowedCrossAttn
from model.heads import RACL

B, T, DV, DA = 4, 50, 16, 24


def test_output_shapes_and_ranges():
    m = RACL(DV, DA).eval()
    o = m(torch.randn(B, T, DV), torch.randn(B, T, DA))
    assert o["c"].shape == (B, T) and o["logit"].shape == (B,)
    assert o["ev"].shape == o["ea"].shape == (B, T, 128)
    assert ((o["c"] > 0) & (o["c"] < 1)).all()
    assert torch.allclose(o["ev"].norm(dim=-1), torch.ones(B, T), atol=1e-5)


def test_attention_window_is_respected():
    att = WindowedCrossAttn(d=32, heads=4, w=5).eval()
    xq, xkv = torch.randn(1, T, 32), torch.randn(1, T, 32)
    base = att(xq, xkv)
    t = 25
    far = xkv.clone(); far[0, t + 6] += 10                  # outside +-5: must not change output at t
    near = xkv.clone(); near[0, t + 5] += 10                # inside the window: must change it
    assert torch.allclose(att(xq, far)[0, t], base[0, t], atol=1e-6)
    assert not torch.allclose(att(xq, near)[0, t], base[0, t], atol=1e-4)


def test_all_parameters_receive_gradient_including_the_bias():
    m = RACL(DV, DA)
    o = m(torch.randn(B, T, DV), torch.randn(B, T, DA))
    (o["c"].mean() + o["logit"].mean() + o["ev"].mean() + o["ea"].mean()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    assert m.att_va.bias.grad.abs().sum() > 0


def test_can_overfit_20_clips():
    torch.manual_seed(0)
    n = 20
    y = (torch.arange(n) % 2).float()                                # half real, half fake
    fv = torch.randn(n, T, DV) + 0.5 * y[:, None, None]              # class-dependent offset
    fa = torch.randn(n, T, DA)
    m = RACL(DV, DA, d=64, layers=1)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    for _ in range(150):
        loss = F.binary_cross_entropy_with_logits(m(fv, fa)["logit"], y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    pred = (m(fv, fa)["logit"] > 0).float()
    assert loss.item() < 0.1 and (pred == y).all()
