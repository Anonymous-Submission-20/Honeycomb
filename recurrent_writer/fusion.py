import torch
import torch.nn as nn

EPS = 1e-6

FUSION_HIDDEN = 48


def _identity_of(multiplicative: bool) -> float:
    return 1.0 if multiplicative else 0.0


def pool_fuse(old, new, w_old, w_new, multiplicative: bool = False):
    ident = _identity_of(multiplicative)
    o = old - ident
    n = new - ident
    total = w_old + w_new
    # avoid dividing by zero without changing cells that have evidence
    denom = torch.where(total > 0, total, torch.ones_like(total))
    pooled = ident + (w_old * o + w_new * n) / denom
    # keep the old value when neither plane has evidence
    return torch.where(total > 0, pooled, old), total


class FusionNet(nn.Module):
    # learn a residual correction from the old, new and pooled planes and their confidences

    def __init__(self, rank: int, multiplicative: bool = False,
                 hidden: int = FUSION_HIDDEN):
        super().__init__()
        self.multiplicative = bool(multiplicative)
        in_ch = 3 * rank + 2
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, rank, 1)
        # start with plain pooling; the network learns a correction
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, old, new, pooled, w_old, w_new):
        x = torch.cat([old, new, pooled,
                       torch.log1p(w_old.clamp_min(0.0)),
                       torch.log1p(w_new.clamp_min(0.0))], dim=1)
        return pooled + self.head(self.body(x))


def fuse_planes(old, new, w_old, w_new, multiplicative: bool = False, net=None):
    pooled, w = pool_fuse(old, new, w_old, w_new, multiplicative=multiplicative)
    if net is None:
        return pooled, w
    return net(old, new, pooled, w_old, w_new), w
