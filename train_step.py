"""Step 7: one training step. The five variants differ ONLY in cfg.flags (Table 10).
Returns (loss, parts); parts holds each term's value (None when the term is off) for logging.
"""
import torch
from losses import cls_loss, nce_loss, pair_loss


def sim(ev, ea):                                         # per-frame cosine similarity [B, T]
    return (ev * ea).sum(-1)


def training_step(model, b, cfg):
    f, L = cfg.flags, cfg.loss
    y, real = b["y"], b["y"] == 0
    oc = model(b["v_clean"], b["a_clean"])
    need_deg = f.degraded_cls or f.nce_deg_pos or f.pair_loss
    od = model(b["v_deg"], b["a_deg"]) if need_deg else None      # C never touches degraded data

    # L_cls: clean (+ degraded) clips, labels unchanged. Desynchronised clips are NOT included.
    logits, ys = oc["logit"], y
    if f.degraded_cls:
        logits, ys = torch.cat([logits, od["logit"]]), torch.cat([y, y])
    l_cls = cls_loss(logits, ys)
    loss, parts = l_cls, {"cls": l_cls.item(), "nce": None, "pair": None}

    # L_NCE: real clips only. Positives: clean pairs (+ degraded pairs if F1). Negatives: desync pairs.
    if real.any():
        os_ = model(b["v_desync"][real], b["a_desync"][real])     # desync views of real clips
        s_pos = [sim(oc["ev"], oc["ea"])[real]]
        if f.nce_deg_pos:
            s_pos.append(sim(od["ev"], od["ea"])[real])
        s_pos = torch.stack(s_pos, -1).flatten(0, 1)              # [(n_real*T), Np]
        s_neg = sim(os_["ev"], os_["ea"]).flatten().unsqueeze(-1)  # [(n_real*T), 1]
        l_nce = nce_loss(s_pos, s_neg, L.tau)
        loss, parts["nce"] = loss + L.lambda1 * l_nce, l_nce.item()

    if f.pair_loss:                                               # L_pair: real clips only
        l_pair = pair_loss(oc["c"], od["c"], real)
        loss, parts["pair"] = loss + L.lambda2 * l_pair, l_pair.item()
    return loss, parts
