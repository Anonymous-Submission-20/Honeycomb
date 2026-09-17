from dataclasses import dataclass

import torch


def incoming_xyz_bounds_from_steps(steps, pts, ok, margin_frac=0.02):
    if not steps:
        raise ValueError("cannot build incoming bounds from empty step set")
    chunks = [pts[k][ok[k]] for k in steps if int(ok[k].sum()) > 0]
    if not chunks:
        raise ValueError("cannot build incoming bounds from steps with no valid points")
    p = torch.cat(chunks, dim=0).detach().cpu().float()
    lo_xyz, hi_xyz = p.amin(dim=0), p.amax(dim=0)
    span = hi_xyz - lo_xyz
    pad = torch.where(span > 1e-6, float(margin_frac) * span, torch.ones_like(span))
    return lo_xyz - pad, hi_xyz + pad

def fixed_clip_time_bounds(K):
    return 0.0, float(int(K) - 1)

def incoming_xyz_inside_bounds(bounds, inc_lo_xyz, inc_hi_xyz, eps=1e-6):
    inc_lo = torch.as_tensor(inc_lo_xyz, dtype=torch.float32).reshape(3)
    inc_hi = torch.as_tensor(inc_hi_xyz, dtype=torch.float32).reshape(3)
    lo = bounds.lo[:3].detach().cpu().float()
    hi = bounds.hi[:3].detach().cpu().float()
    return bool(torch.all(inc_lo >= lo - float(eps)) and torch.all(inc_hi <= hi + float(eps)))

def make_reserved_xyz_fixedt_bounds(current_bounds, inc_lo_xyz, inc_hi_xyz, K,
                                    growth_factor=1.5, needed_factor=1.1,
                                    eps=1e-6):
    inc_lo = torch.as_tensor(inc_lo_xyz, dtype=torch.float32).reshape(3)
    inc_hi = torch.as_tensor(inc_hi_xyz, dtype=torch.float32).reshape(3)
    new_lo = current_bounds.lo.detach().cpu().float().clone()
    new_hi = current_bounds.hi.detach().cpu().float().clone()
    cur_lo = current_bounds.lo[:3].detach().cpu().float()
    cur_hi = current_bounds.hi[:3].detach().cpu().float()
    for axis in range(3):
        below = bool(inc_lo[axis] < cur_lo[axis] - float(eps))
        above = bool(inc_hi[axis] > cur_hi[axis] + float(eps))
        if not (below or above):
            continue
        needed_lo = torch.minimum(cur_lo[axis], inc_lo[axis])
        needed_hi = torch.maximum(cur_hi[axis], inc_hi[axis])
        cur_span = (cur_hi[axis] - cur_lo[axis]).clamp_min(1e-8)
        needed_span = (needed_hi - needed_lo).clamp_min(1e-8)
        target_span = torch.maximum(
            torch.as_tensor(float(growth_factor)) * cur_span,
            torch.as_tensor(float(needed_factor)) * needed_span,
        )
        if above and not below:
            new_lo[axis] = cur_lo[axis]
            new_hi[axis] = cur_lo[axis] + target_span
        elif below and not above:
            new_hi[axis] = cur_hi[axis]
            new_lo[axis] = cur_hi[axis] - target_span
        else:
            center = 0.5 * (needed_lo + needed_hi)
            new_lo[axis] = center - 0.5 * target_span
            new_hi[axis] = center + 0.5 * target_span
    t_lo, t_hi = fixed_clip_time_bounds(K)
    new_lo[3] = t_lo
    new_hi[3] = t_hi
    return AdaptiveBounds(lo=new_lo, hi=new_hi)

@dataclass
class AdaptiveBounds:
    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self):
        self.lo = torch.as_tensor(self.lo, dtype=torch.float32).reshape(4)
        self.hi = torch.as_tensor(self.hi, dtype=torch.float32).reshape(4)
        if torch.any(self.hi <= self.lo):
            raise ValueError("AdaptiveBounds requires hi > lo for all axes")

    @classmethod
    def from_steps(cls, steps, pts, ok, margin_frac=0.02, time_margin=0.5,
                   fixed_time_bounds=None):
        if not steps:
            raise ValueError("cannot build adaptive bounds from empty step set")
        lo_xyz, hi_xyz = incoming_xyz_bounds_from_steps(
            steps, pts, ok, margin_frac=margin_frac)
        if fixed_time_bounds is None:
            t_lo = float(min(steps)) - float(time_margin)
            t_hi = float(max(steps)) + float(time_margin)
            if t_hi <= t_lo:
                t_hi = t_lo + 1.0
        else:
            t_lo, t_hi = [float(x) for x in fixed_time_bounds]
        lo = torch.cat([lo_xyz, torch.tensor([t_lo])])
        hi = torch.cat([hi_xyz, torch.tensor([t_hi])])
        return cls(lo=lo, hi=hi)

    def expand_to_steps(self, steps, pts, ok, margin_frac=0.02, time_margin=0.5,
                        fixed_time_bounds=None):
        incoming = AdaptiveBounds.from_steps(
            steps, pts, ok, margin_frac=margin_frac, time_margin=time_margin,
            fixed_time_bounds=fixed_time_bounds)
        return AdaptiveBounds(lo=torch.minimum(self.lo, incoming.lo),
                              hi=torch.maximum(self.hi, incoming.hi))

    def span(self):
        return self.hi - self.lo

    def spatial_extents(self):
        return self.span()[:3].tolist()

    def world_to_norm4(self, q4, clamp=True):
        if torch.is_tensor(q4):
            q4 = q4.to(dtype=torch.float32)
        else:
            q4 = torch.as_tensor(q4, dtype=torch.float32, device=self.lo.device)
        lo = self.lo.to(q4.device)
        hi = self.hi.to(q4.device)
        out = 2.0 * (q4 - lo) / (hi - lo).clamp_min(1e-8) - 1.0
        return out.clamp(-1.0, 1.0) if clamp else out

    def world_to_norm(self, p_world, tau, clamp=True):
        if tau.ndim == 0:
            tau = tau.expand(p_world.shape[0])
        q4 = torch.cat([p_world, tau.reshape(-1, 1).to(p_world.device, p_world.dtype)], dim=1)
        qn = self.world_to_norm4(q4, clamp=clamp)
        return qn[:, :3], qn[:, 3]

    def to_json(self):
        return {"lo": [float(x) for x in self.lo.tolist()],
                "hi": [float(x) for x in self.hi.tolist()]}
