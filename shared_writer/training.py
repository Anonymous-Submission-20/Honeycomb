import time

import torch

from shared_writer.model import functional_forward_ray


def to_dev(clip, device):
    out = {"clip_id": clip["clip_id"], "K": clip["K"],
           "write_t": clip["write_t"],
           "mean": clip["mean"].to(device), "std": clip["std"].to(device)}
    out["write"] = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in clip["write"].items()}
    out["q"] = {k: v.to(device) for k, v in clip["q"].items()}
    return out


def run_clip(writer, clip, subsample=None):
    w, q = clip["write"], clip["q"]
    spatial, st = writer.write(w["p"], w["t"], w["F"], w["VD"], w["ORG"],
                               w["extents"], w["t_res"])
    n = q["p"].shape[0]
    if subsample is not None and subsample < n:
        sel = torch.randint(0, n, (subsample,), device=q["p"].device)
    else:
        sel = slice(None)
    pred_n = functional_forward_ray(
        spatial, st, writer.reader, q["p"][sel], q["vd"][sel], q["org"][sel],
        q["t"][sel])
    return ((pred_n - q["mem_norm"][sel]) ** 2).mean()


def corr(a, b):
    # compute this metric on CPU because MPS does not support float64
    a = a.reshape(-1).detach().cpu().double()
    b = b.reshape(-1).detach().cpu().double()
    return float(((a - a.mean()) * (b - b.mean())).mean()
                 / (a.std() * b.std()).clamp_min(1e-8))


@torch.no_grad()
def evaluate(writer, clips, device):
    writer.eval()
    rows = []
    for clip in clips:
        c = to_dev(clip, device)
        t0 = time.time()
        w = c["write"]
        spatial, st = writer.write(w["p"], w["t"], w["F"], w["VD"], w["ORG"],
                                   w["extents"], w["t_res"])
        if device.type in ("cuda", "mps"):
            torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
        wall_write = time.time() - t0
        q = c["q"]
        pred_n = functional_forward_ray(
            spatial, st, writer.reader, q["p"], q["vd"], q["org"], q["t"])
        pred_raw = pred_n * c["std"] + c["mean"]

        tgt_t = clip["write_t"]
        t_half = q["frame"] >= tgt_t
        mse = lambda a, b: float(((a - b) ** 2).mean())  # noqa: E731
        rows.append({
            "clip_id": c["clip_id"],
            "n_queries": int(q["p"].shape[0]),
            "mse_norm": float(((pred_n - q["mem_norm"]) ** 2).mean()),
            "retr_mse_writer_T": mse(pred_raw[t_half], q["mem_raw"][t_half]),
            "retr_corr_writer_T": corr(pred_raw[t_half], q["mem_raw"][t_half]),
            "std_ratio_vs_memory": float(
                pred_raw.std() / q["mem_raw"].std().clamp_min(1e-8)),
            "wall_write_s": round(wall_write, 4),
        })
    writer.train()
    return rows


def med(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def rnd(v, nd):
    return None if v is None else round(v, nd)
