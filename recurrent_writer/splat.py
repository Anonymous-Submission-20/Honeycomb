from .frame import IncrementalFrame
from .warp import PLANE_SPECS


def resolutions_from_shapes(plane_shapes: dict) -> dict:
    res = {}
    for i, ((aw, ah), _ac) in enumerate(PLANE_SPECS):
        _, _, h, w = plane_shapes["spatial"][i]
        res[aw] = w
        res[ah] = h
    missing = {"x", "y", "z"} - set(res)
    if missing:
        raise ValueError(f"could not recover resolutions for axes {sorted(missing)}")
    return res


def normalise_chunk(chunk, frame: IncrementalFrame):
    for key in ("points_world", "times", "feats", "viewdirs", "origins_world"):
        if key not in chunk:
            raise ValueError(f"chunk missing {key!r}; origins must be WORLD-space")
    return (frame.to_norm_xyz(chunk["points_world"]),
            frame.to_norm_t(chunk["times"]),
            frame.to_norm_xyz(chunk["origins_world"]))


def splat_chunk(writer, chunk, frame: IncrementalFrame, plane_shapes: dict):
    res = resolutions_from_shapes(plane_shapes)
    t_res = plane_shapes["st"][0][2]
    p_norm, t_norm, org_norm = normalise_chunk(chunk, frame)
    return writer.write(p_norm, t_norm, chunk["feats"], chunk["viewdirs"],
                        org_norm, extents=None, t_res=t_res, res=res,
                        return_density=True)


def initial_state_planes(writer, chunk, frame: IncrementalFrame, extents, t_res):
    # choose plane shapes once from the seed; later writes keep those shapes
    p_norm, t_norm, org_norm = normalise_chunk(chunk, frame)
    return writer.write(p_norm, t_norm, chunk["feats"], chunk["viewdirs"],
                        org_norm, extents=extents, t_res=t_res,
                        return_density=True)
