"""Step 4: RACL assembly, Eq. (2): c(t) = sigmoid( g([V_t; A_t; Attn_v->a(t); Attn_a->v(t)]) ).
Hidden layout of g, the pooled classifier input (D-08) and the NCE projections (D-09) are [Proposed].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.windowed_attn import WindowedCrossAttn


class RACL(nn.Module):
    def __init__(self, dv, da, d=256, layers=2, heads=4, w=5, bias_range=10, proj_dim=128):
        super().__init__()
        self.pv, self.pa = nn.Linear(dv, d), nn.Linear(da, d)          # input projections from encoder dims
        mk = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.tv, self.ta = mk(), mk()                                   # trainable temporal Transformers -> V, A
        self.att_va = WindowedCrossAttn(d, heads, w, bias_range)        # video queries audio
        self.att_av = WindowedCrossAttn(d, heads, w, bias_range)        # audio queries video
        self.g = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d // 2), nn.GELU())
        self.score = nn.Linear(d // 2, 1)                               # consistency head -> c(t)
        self.cls = nn.Linear(d // 2, 1)                                 # clip classifier on pooled features (D-08)
        self.proj_v, self.proj_a = nn.Linear(d, proj_dim), nn.Linear(d, proj_dim)   # NCE embeddings (D-09)

    def forward(self, fv, fa):                                          # frozen cached features [B, T, D*]
        V, A = self.tv(self.pv(fv)), self.ta(self.pa(fa))
        h = self.g(torch.cat([V, A, self.att_va(V, A), self.att_av(A, V)], -1))
        return {"c": torch.sigmoid(self.score(h)).squeeze(-1),          # [B, T], high = consistent
                "logit": self.cls(h.mean(1)).squeeze(-1),               # > 0 -> fake (y = 1)
                "ev": F.normalize(self.proj_v(V), dim=-1),
                "ea": F.normalize(self.proj_a(A), dim=-1)}
