import torch
import torch.nn.functional as Fn

AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "t": 3}

# each pair lists width and height axes, followed by the spatiotemporal spatial axis
PLANE_SPECS = [(("x", "y"), "z"), (("x", "z"), "y"), (("y", "z"), "x")]

SPATIAL_EMPTY = 0.0
ST_EMPTY = 1.0

_BOUNDS_EPS = 1e-8


def _world_from_norm(norm, bounds, axis, time_map):
    idx = AXIS_INDEX[axis]
    lo = float(bounds.lo[idx])
    hi = float(bounds.hi[idx])
    if axis == "t":
        return time_map.from_norm(norm, lo, hi)
    return 0.5 * (norm + 1.0) * (hi - lo) + lo


def _norm_from_world(world, bounds, axis, time_map):
    idx = AXIS_INDEX[axis]
    lo = float(bounds.lo[idx])
    hi = float(bounds.hi[idx])
    if axis == "t":
        return time_map.to_norm(world, lo, hi)
    return 2.0 * (world - lo) / max(hi - lo, 1e-8) - 1.0


def axis_moved(old_bounds, new_bounds, axis, eps: float = _BOUNDS_EPS) -> bool:
    idx = AXIS_INDEX[axis]
    return bool(abs(float(old_bounds.lo[idx]) - float(new_bounds.lo[idx])) > eps
                or abs(float(old_bounds.hi[idx]) - float(new_bounds.hi[idx])) > eps)


def _inside_old(world, bounds, axis, eps_frac: float = 1e-6):
    idx = AXIS_INDEX[axis]
    lo = float(bounds.lo[idx])
    hi = float(bounds.hi[idx])
    eps = eps_frac * max(hi - lo, 1e-8)
    return (world >= lo - eps) & (world <= hi + eps)


def warp_plane(plane, axes, old_bounds, new_bounds, time_map,
               fill_value: float = SPATIAL_EMPTY):
    w_axis, h_axis = axes
    if not (axis_moved(old_bounds, new_bounds, w_axis)
            or axis_moved(old_bounds, new_bounds, h_axis)):
        return plane

    _, _, h, w = plane.shape
    dev = plane.device
    compute_dtype = (torch.float32
                     if plane.dtype in (torch.bfloat16, torch.float16)
                     else plane.dtype)
    work = plane.to(compute_dtype)

    dst_w = torch.linspace(-1.0, 1.0, w, device=dev, dtype=compute_dtype)
    dst_h = torch.linspace(-1.0, 1.0, h, device=dev, dtype=compute_dtype)
    vv, uu = torch.meshgrid(dst_h, dst_w, indexing="ij")

    world_w = _world_from_norm(uu, new_bounds, w_axis, time_map)
    world_h = _world_from_norm(vv, new_bounds, h_axis, time_map)

    # check the original bounds before clamping coordinates
    valid = (_inside_old(world_w, old_bounds, w_axis)
             & _inside_old(world_h, old_bounds, h_axis)).view(1, 1, h, w)

    src_w = _norm_from_world(world_w, old_bounds, w_axis, time_map)
    src_h = _norm_from_world(world_h, old_bounds, h_axis, time_map)

    grid = torch.stack([src_w, src_h], dim=-1).unsqueeze(0)
    warped = Fn.grid_sample(work, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    # fill new regions explicitly because clamping prevents grid_sample padding
    warped = torch.where(valid, warped,
                         torch.full_like(warped, float(fill_value)))
    return warped.to(plane.dtype)


def warp_field(spatial, st, old_bounds, new_bounds, time_map,
               spatial_fill: float = SPATIAL_EMPTY, st_fill: float = ST_EMPTY):
    if len(spatial) != len(PLANE_SPECS) or len(st) != len(PLANE_SPECS):
        raise ValueError(
            f"expected {len(PLANE_SPECS)} spatial and st planes, "
            f"got {len(spatial)} and {len(st)}")
    out_sp, out_st = [], []
    for i, ((aw, ah), ac) in enumerate(PLANE_SPECS):
        out_sp.append(warp_plane(spatial[i], (aw, ah), old_bounds, new_bounds,
                                 time_map, fill_value=spatial_fill))
        out_st.append(warp_plane(st[i], (ac, "t"), old_bounds, new_bounds,
                                 time_map, fill_value=st_fill))
    return out_sp, out_st


def warp_weights(w_spatial, w_st, old_bounds, new_bounds, time_map):
    # warp confidence with the planes and give new regions zero confidence
    return warp_field(w_spatial, w_st, old_bounds, new_bounds, time_map,
                      spatial_fill=0.0, st_fill=0.0)
