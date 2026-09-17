import torch

from .frame import IncrementalFrame


def _world_from_norm_xyz(norm, lo, hi):
    return 0.5 * (norm + 1.0) * (hi[:3] - lo[:3]) + lo[:3]


def pack_chunks(pack, rung_for_geometry: int = 0):
    p = (pack if isinstance(pack, dict)
         else torch.load(pack, map_location="cpu", weights_only=False))
    write_sets = p["write_sets"]
    valid = p["valid"].bool()
    frame_of = p["frame_of"].long()
    remap = torch.cumsum(valid.long(), 0) - 1
    F_all, VD_all = p["F"].float(), p["VD"].float()

    g = int(rung_for_geometry)
    lo, hi = p["bounds_lo"][g], p["bounds_hi"][g]
    org_world_per_frame = _world_from_norm_xyz(
        p["origins_frame"][g].float(), lo, hi)

    chunks, seen = [], set()
    for ws in write_sets:
        new_times = sorted(set(ws) - seen)
        seen |= set(ws)
        if not new_times:
            continue
        sel = torch.zeros_like(valid)
        for i in new_times:
            sel |= (frame_of == i)
        mask = valid & sel
        if not bool(mask.any()):
            raise ValueError(f"{p['clip_id']}: chunk {new_times} has no valid points")
        rows = remap[mask]
        chunks.append({
            "points_world": p["points_world"][mask],
            # use the latent-frame index as world time
            "times": frame_of[mask].float(),
            "feats": F_all[rows],
            "viewdirs": VD_all[rows],
            "origins_world": org_world_per_frame[frame_of[mask]].contiguous(),
            "latent_times": new_times,
        })
    return chunks


def pack_queries(pack, rung: int, rung_for_geometry: int = 0):
    p = (pack if isinstance(pack, dict)
         else torch.load(pack, map_location="cpu", weights_only=False))
    valid = p["valid"].bool()
    frame_of = p["frame_of"].long()
    remap = torch.cumsum(valid.long(), 0) - 1
    F_all = p["F"].float()
    g = int(rung_for_geometry)
    lo, hi = p["bounds_lo"][g], p["bounds_hi"][g]
    org_world = _world_from_norm_xyz(p["origins_frame"][g].float(), lo, hi)

    sel_r, hit_r = p["sel"][rung], p["hit"][rung]
    dirs = p["dirs"]
    qp, qvd, qorg, qt, mem_n, qframe = [], [], [], [], [], []
    for k in range(int(p["num_real_times"])):
        cells = torch.nonzero(hit_r[k], as_tuple=False).squeeze(1)
        if cells.numel() == 0:
            continue
        idx = sel_r[k, cells].long()
        qp.append(p["points_world"][idx])
        qvd.append(dirs[k, cells].float())
        qorg.append(org_world[k].reshape(1, 3).expand(cells.numel(), 3))
        qt.append(frame_of[idx].float())
        mem_n.append(F_all[remap[idx]])
        qframe.append(torch.full((cells.numel(),), k, dtype=torch.long))
    mem = torch.cat(mem_n)
    return {"points_world": torch.cat(qp),
            "viewdirs": torch.cat(qvd),
            "origins_world": torch.cat(qorg).contiguous(),
            "times": torch.cat(qt),
            "mem_norm": mem,
            "mem_raw": mem * p["std"] + p["mean"],
            # use the clip standard deviation to report error in raw latent units
            "std": p["std"].clone(),
            "frame": torch.cat(qframe)}


def initial_frame(pack, time_map, chunk_span: float = 8.0,
                  rung_for_geometry: int = 0) -> IncrementalFrame:
    # use the seed spatial bounds and one chunk of time
    p = (pack if isinstance(pack, dict)
         else torch.load(pack, map_location="cpu", weights_only=False))
    g = int(rung_for_geometry)
    from honeycomb import AdaptiveBounds
    lo = p["bounds_lo"][g].clone().float()
    hi = p["bounds_hi"][g].clone().float()
    lo[3], hi[3] = 0.0, float(chunk_span)
    return IncrementalFrame(AdaptiveBounds(lo=lo, hi=hi), time_map)


def run_recurrence(writer, chunks, frame, t_res: int = 25):
    state, states = None, []
    for c in chunks:
        state = writer.update(state, c, frame=frame if state is None else None,
                              t_res=t_res)
        states.append(state)
    return states
