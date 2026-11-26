"""Step 4: windowed cross-modal attention, Eq. (1):
Attn = softmax( Q K^T / sqrt(d_k) + B(dt) + M_w ) V
Per-head sqrt(d_k) is used instead of the paper's sqrt(d) (C-02, D-07). Bias range +-10 vs window +-5: D-07.
"""
import math
import torch
import torch.nn as nn


class WindowedCrossAttn(nn.Module):
    def __init__(self, d=256, heads=4, w=5, bias_range=10):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk, self.w, self.r = heads, d // heads, w, bias_range
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.bias = nn.Parameter(torch.zeros(heads, 2 * bias_range + 1))   # B(dt), dt in [-r, r]

    def forward(self, xq, xkv):                                            # [B, T, d] each
        B, T, _ = xq.shape
        sp = lambda z: z.view(B, T, self.h, self.dk).transpose(1, 2)       # [B, h, T, dk]
        q, k, v = sp(self.q(xq)), sp(self.k(xkv)), sp(self.v(xkv))
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)              # [B, h, T, T]
        idx = torch.arange(T, device=xq.device)
        dt = idx[None, :] - idx[:, None]                                   # key frame minus query frame
        logits = logits + self.bias[:, dt.clamp(-self.r, self.r) + self.r]
        logits = logits.masked_fill((dt.abs() > self.w)[None, None], float("-inf"))   # M_w
        out = (logits.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o(out)
