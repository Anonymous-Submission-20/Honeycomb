import torch


def rung_count(pack_or_path):
    p = (pack_or_path if isinstance(pack_or_path, dict)
         else torch.load(pack_or_path, map_location="cpu", weights_only=False))
    return len(p["write_sets"])


def load_pack_rollout(path, rung=-1):
    # load one write-count rung: 1, 9 or 17 latent frames; rung 0 is the seed
    p = (path if isinstance(path, dict)
         else torch.load(path, map_location="cpu", weights_only=False))
    n_real = int(p["num_real_times"])
    n_slots = int(p["num_time_steps"])
    if n_real > n_slots:
        raise ValueError(
            f"{p['clip_id']}: {n_real} real times exceed {n_slots} temporal slots")

    write_sets = p["write_sets"]
    rung = range(len(write_sets))[rung]
    ws = write_sets[rung]

    lo, hi = p["bounds_lo"][rung], p["bounds_hi"][rung]
    span3 = (hi[:3] - lo[:3]).clamp_min(1e-8)
    pts_norm = (2.0 * (p["points_world"] - lo[:3]) / span3 - 1.0).clamp(-1, 1)

    valid = p["valid"].bool()
    frame_of = p["frame_of"].long()
    tau_frame = (2.0 * torch.arange(n_real, dtype=torch.float32)
                 / max(n_slots - 1, 1) - 1.0)

    in_set = torch.zeros_like(valid)
    for i in ws:
        in_set |= (frame_of == i)
    w_mask = valid & in_set
    if not bool(w_mask.any()):
        raise ValueError(f"{p['clip_id']}: rung {rung} writes no valid points")

    remap = torch.cumsum(valid.long(), 0) - 1
    F_all, VD_all = p["F"].float(), p["VD"].float()
    rows_w = remap[w_mask]

    origins_frame = p["origins_frame"][rung].float()
    write = {
        "p": pts_norm[w_mask],
        "t": tau_frame[frame_of[w_mask]],
        "F": F_all[rows_w],
        "VD": VD_all[rows_w],
        "ORG": origins_frame[frame_of[w_mask]].contiguous(),
        "extents": (hi[:3] - lo[:3]).tolist(),
        "t_res": n_slots,
    }

    sel_r, hit_r = p["sel"][rung], p["hit"][rung]
    dirs = p["dirs"]
    qp, qvd, qorg, qt, mem_n, frame_of_q = [], [], [], [], [], []
    for k in range(n_real):
        cells = torch.nonzero(hit_r[k], as_tuple=False).squeeze(1)
        if cells.numel() == 0:
            continue
        idx = sel_r[k, cells].long()
        if not bool(w_mask[idx].all()):
            raise ValueError(
                f"{p['clip_id']}: rung {rung} frame {k} selected a point that "
                f"was not written -- sel and the write set disagree")
        qp.append(pts_norm[idx])
        qvd.append(dirs[k, cells].float())
        qorg.append(origins_frame[k].reshape(1, 3).expand(cells.numel(), 3))
        qt.append(tau_frame[frame_of[idx]])
        mem_n.append(F_all[remap[idx]])
        frame_of_q.append(torch.full((cells.numel(),), k, dtype=torch.long))

    mem = torch.cat(mem_n)
    mean, std = p["mean"], p["std"]
    q = {
        "p": torch.cat(qp),
        "vd": torch.cat(qvd),
        "org": torch.cat(qorg).contiguous(),
        "t": torch.cat(qt),
        "mem_norm": mem,
        "mem_raw": mem * std + mean,
        "frame": torch.cat(frame_of_q),
    }
    return {"clip_id": p["clip_id"], "write": write, "q": q,
            "num_real_times": n_real, "num_time_steps": n_slots,
            # report the full rollout grid size, including padded slots
            "K": n_slots,
            "write_t": int(p["write_t"]), "rung": rung, "write_set": list(ws),
            "mean": mean, "std": std}
