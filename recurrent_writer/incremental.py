import torch
import torch.nn as nn

from honeycomb import AdaptiveBounds
from honeycomb.adaptive import (
    incoming_xyz_bounds_from_steps, incoming_xyz_inside_bounds,
    make_reserved_xyz_fixedt_bounds)
from shared_writer.model import functional_forward_ray

from .frame import IncrementalFrame
from .fusion import FUSION_HIDDEN, FusionNet, fuse_planes
from .splat import initial_state_planes, normalise_chunk, splat_chunk
from .state import IncrementalState

_EPS = 1e-6
_MAX_EXPANSIONS = 1024          # stop runaway growth from invalid timestamps

BOUNDS_POLICY = "tight"


class IncrementalWriter(nn.Module):
    def __init__(self, writer, fusion: str = "pool", hidden: int = FUSION_HIDDEN,
                 bounds_policy: str = BOUNDS_POLICY):
        super().__init__()
        if bounds_policy not in ("tight", "reserve"):
            raise ValueError(f"bounds_policy must be 'tight' or 'reserve', got {bounds_policy!r}")
        self.bounds_policy = bounds_policy
        if fusion not in ("pool", "learned"):
            raise ValueError(f"fusion must be 'pool' or 'learned', got {fusion!r}")
        self.writer = writer
        self.fusion = fusion
        if fusion == "learned":
            self.net_spatial = nn.ModuleList(
                [FusionNet(rank=r, multiplicative=False, hidden=hidden)
                 for r in writer.ranks])
            self.net_st = nn.ModuleList(
                [FusionNet(rank=r, multiplicative=True, hidden=hidden)
                 for r in writer.ranks])
        else:
            self.net_spatial = self.net_st = None

    @property
    def max_res(self):
        return self.writer.max_res

    def grown_frame(self, frame: IncrementalFrame, chunk,
                    margin_frac: float = 0.02) -> IncrementalFrame:
        tmax = float(chunk["times"].max())
        out = frame
        for _ in range(_MAX_EXPANSIONS):
            if tmax <= out.t_hi + _EPS:
                break
            out = out.expand_time_by_chunk()
        else:
            raise RuntimeError(
                f"time bounds did not reach {tmax} after {_MAX_EXPANSIONS} chunks")

        pts = chunk["points_world"].detach().cpu().float()
        inc_lo, inc_hi = incoming_xyz_bounds_from_steps(
            [0], pts.unsqueeze(0),
            torch.ones(1, pts.shape[0], dtype=torch.bool), margin_frac=margin_frac)

        if incoming_xyz_inside_bounds(out.bounds, inc_lo, inc_hi):
            return out

        cur_lo, cur_hi = out.bounds.lo.clone(), out.bounds.hi.clone()
        if self.bounds_policy == "tight":
            cur_lo[:3] = torch.minimum(cur_lo[:3], inc_lo)
            cur_hi[:3] = torch.maximum(cur_hi[:3], inc_hi)
            return IncrementalFrame(AdaptiveBounds(lo=cur_lo, hi=cur_hi), out.time_map)

        # reserve extra space while keeping the recurrent time axis
        t_lo, t_hi = float(out.bounds.lo[3]), float(out.bounds.hi[3])
        nb = make_reserved_xyz_fixedt_bounds(out.bounds, inc_lo, inc_hi, K=2)
        lo, hi = nb.lo.clone(), nb.hi.clone()
        lo[3], hi[3] = t_lo, t_hi
        return IncrementalFrame(AdaptiveBounds(lo=lo, hi=hi), out.time_map)

    def update(self, state, chunk, frame: IncrementalFrame = None,
               t_res: int = 25) -> IncrementalState:
        if state is None:
            if frame is None:
                raise ValueError("the first write needs an initial frame")
            grown = self.grown_frame(frame, chunk)
            sp, st, d_sp, d_st = initial_state_planes(
                self.writer, chunk, grown,
                extents=grown.bounds.spatial_extents(), t_res=t_res)
            return IncrementalState(sp, st, d_sp, d_st, grown)

        new_frame = self.grown_frame(state.frame, chunk)
        state.frame.assert_compatible(new_frame)
        warped = state.warped_to(new_frame)

        cand_sp, cand_st, d_sp, d_st = splat_chunk(
            self.writer, chunk, new_frame, state.plane_shapes)
        state.assert_shapes_match(cand_sp, cand_st)

        out_sp, out_st, out_wsp, out_wst = [], [], [], []
        for i in range(len(self.writer.ranks)):
            p, w = fuse_planes(
                warped.spatial[i], cand_sp[i], warped.w_spatial[i], d_sp[i],
                multiplicative=False,
                net=None if self.net_spatial is None else self.net_spatial[i])
            out_sp.append(p)
            out_wsp.append(w)
            p, w = fuse_planes(
                warped.st[i], cand_st[i], warped.w_st[i], d_st[i],
                multiplicative=True,
                net=None if self.net_st is None else self.net_st[i])
            out_st.append(p)
            out_wst.append(w)
        return IncrementalState(out_sp, out_st, out_wsp, out_wst, new_frame)

    def query(self, state: IncrementalState, points_world, times_world,
              viewdirs, origins_world):
        p_norm, t_norm, org_norm = normalise_chunk(
            {"points_world": points_world, "times": times_world,
             "feats": None, "viewdirs": viewdirs,
             "origins_world": origins_world}, state.frame)
        return functional_forward_ray(state.spatial, state.st, self.writer.reader,
                                      p_norm, viewdirs, org_norm, t_norm)


class StateField(nn.Module):
    # the reader receives origins already normalized to the current frame

    def __init__(self, state: IncrementalState, reader):
        super().__init__()
        self.state = state
        self.reader = reader

    @property
    def bounds(self):
        return self.state.frame.bounds

    def forward_ray(self, p_world, viewdir, origin, tau):
        f = self.state.frame
        p_norm = f.to_norm_xyz(p_world).clamp(-1.0, 1.0)
        t_norm = f.to_norm_t(tau)
        return functional_forward_ray(
            self.state.spatial, self.state.st, self.reader,
            p_norm.to(p_world.device), viewdir, origin,
            t_norm.to(p_world.device))


def load_incremental_writer(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if "fusion" not in a:
        raise ValueError(
            f"{ckpt_path} has no 'fusion' in args -- this is a REPLACEMENT writer "
            "checkpoint, not an incremental one")
    from shared_writer.model import PlaneWriter
    base = PlaneWriter(base=a["base"], refine_depth=a["depth"],
                       point_dim=a["point_dim"], contrib_dim=a["contrib_dim"],
                       hidden=a["reader_hidden"], p_in_encoder=a.get("p_input", False),
                       point_layers=a["point_layers"], max_res=a["max_res"],
                       ranks=tuple(a["ranks"]))
    model = IncrementalWriter(base, fusion=a["fusion"],
                              hidden=a.get("fusion_hidden", FUSION_HIDDEN),
                              bounds_policy=a.get("bounds_policy", BOUNDS_POLICY))
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.ckpt_meta = {"path": str(ckpt_path), "step": ck.get("step"),
                       "epoch": ck.get("epoch"), "val_score": ck.get("val_score"),
                       "time_map": a.get("time_map"), "chunk_span": a.get("chunk_span"),
                       "fusion": a["fusion"], "fusion_hidden": a.get("fusion_hidden"),
                       "bounds_policy": a.get("bounds_policy")}
    return model
