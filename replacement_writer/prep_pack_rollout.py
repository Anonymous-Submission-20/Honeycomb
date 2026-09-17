# store 21 real latent times on the 25-slot grid used by a padded rollout

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter.corpus_utils import (  # noqa: E402
    build_readout_rays, cell_rays, indexed_project, normalize_origin,
    scale_intrinsics_to_latent)
from honeycomb import (  # noqa: E402
    AdaptiveBounds, incoming_xyz_bounds_from_steps, incoming_xyz_inside_bounds,
    make_reserved_xyz_fixedt_bounds)
from replacement_writer.rollout_timeline import (  # noqa: E402
    ROLLOUT_RUNGS, latent_frame_indices, rollout_time_bounds,
    rollout_write_sets)

DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_TIME_STEPS = 25          # pad 81 frames to a 97-frame rollout, giving 25 latent slots
MIN_ABS_FOCAL = 50.0


def accumulate(clip, lpc, frames, write_t, LatentPointCloud, latents,
               device="cpu"):
    pts = [lpc.points_world.float()]
    feats = [lpc.features.float()]
    valid = [lpc.valid_mask.bool()]
    frame_of = [torch.full((lpc.points_world.shape[0],), write_t,
                           dtype=torch.long)]

    for i, f in enumerate(frames):
        if i == write_t:
            continue
        fl = LatentPointCloud.from_geometry(
            depth=clip["depths"][f],
            intrinsics=clip["intrinsics"][f],
            cam2world=clip["poses_c2w"][f],
            latent=latents[i].unsqueeze(1),
            mask=None,
            device=device,
        )
        v = fl.valid_mask.bool()
        if not bool(v.any()):
            continue
        pts.append(fl.points_world[v].float())
        feats.append(fl.features[v].float())
        valid.append(torch.ones(int(v.sum()), dtype=torch.bool))
        frame_of.append(torch.full((int(v.sum()),), i, dtype=torch.long))

    return (torch.cat(pts), torch.cat(feats), torch.cat(valid),
            torch.cat(frame_of))


def write_dirs(clip, frames, lat_h, lat_w):
    img_h, img_w = clip["depths"].shape[1:3]
    out = []
    for f in frames:
        K = scale_intrinsics_to_latent(clip["intrinsics"][f], lat_h, lat_w,
                                       img_h, img_w)
        d, _ = cell_rays(clip["poses_c2w"][f], K, lat_h, lat_w)
        out.append(d)
    return torch.stack(out)


def _frame_valid_cells(clip, f, lat_h, lat_w, device):
    d = torch.as_tensor(clip["depths"][f], dtype=torch.float32, device=device)
    d_lat = Fn.interpolate(d[None, None], size=(lat_h, lat_w), mode="bilinear",
                           align_corners=False)[0, 0]
    return torch.nonzero(d_lat.reshape(-1) > 0, as_tuple=False).squeeze(1)


def load_clip_geometry(clip_dir):
    clip_dir = Path(clip_dir)
    g = np.load(clip_dir / "geometry.npz")
    return {
        "clip_id": clip_dir.name,
        "depths": g["depths"],
        "poses_c2w": g["poses_c2w"],
        "intrinsics": g["intrinsics"],
    }


def check_encoder_complete(clips, quiet=False):
    missing = [c.name for c in clips
               if not (c / "clip.pt").exists()
               or (c / "clip.pt").stat().st_size == 0]
    if not quiet:
        print(f"encoder preflight: {len(clips) - len(missing)}/{len(clips)} "
              f"clips have clip.pt"
              + (f"; MISSING e.g. {missing[:5]}" if missing else ""))
    return missing


def _require_finite(name, t, clip_id):
    if not torch.isfinite(t).all():
        raise ValueError(f"{clip_id}: {name} contains non-finite values")


def validate_geometry(clip):
    cid = clip["clip_id"]
    d = torch.as_tensor(np.asarray(clip["depths"]), dtype=torch.float32)
    _require_finite("depths", d, cid)
    if bool((d <= 0).any()):
        raise ValueError(f"{cid}: depths contain non-positive values")
    K = torch.as_tensor(np.asarray(clip["intrinsics"]), dtype=torch.float32)
    _require_finite("intrinsics", K, cid)
    # negative focal lengths are valid in ViPE; reject only implausible magnitudes
    fx, fy = K[:, 0, 0].abs(), K[:, 1, 1].abs()
    if bool((fx < MIN_ABS_FOCAL).any()) or bool((fy < MIN_ABS_FOCAL).any()):
        raise ValueError(
            f"{cid}: implausible focal magnitude (min |fx| {float(fx.min()):.2f}, "
            f"|fy| {float(fy.min()):.2f}; threshold {MIN_ABS_FOCAL})")
    _require_finite("poses_c2w",
                    torch.as_tensor(np.asarray(clip["poses_c2w"]), dtype=torch.float32), cid)


def full_clip_latents(clip_dir, n_real):
    path = Path(clip_dir) / "clip.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `run_video_vae_encode --video-keys clip` first")
    raw = torch.load(path, map_location="cpu", weights_only=False)["latent"].float()
    if raw.shape[0] != n_real:
        raise ValueError(
            f"{path}: {raw.shape[0]} latents but {n_real} real times expected")
    return {i: raw[i] for i in range(n_real)}


def build_rollout_pack(src_dir, LatentPointCloud, num_frames=DEFAULT_NUM_FRAMES,
                       num_time_steps=DEFAULT_NUM_TIME_STEPS,
                       rungs=ROLLOUT_RUNGS, device="cpu"):
    src_dir = Path(src_dir)
    clip = load_clip_geometry(src_dir)
    if clip["depths"].shape[0] < num_frames:
        raise ValueError(
            f"{src_dir.name}: geometry has {clip['depths'].shape[0]} frames, "
            f"need {num_frames}")

    validate_geometry(clip)
    frames = latent_frame_indices(num_frames)
    n_real = len(frames)
    write_t = 0
    latents = full_clip_latents(src_dir, n_real)
    lat_c, lat_h, lat_w = latents[0].shape
    n_cells = lat_h * lat_w

    lpc = LatentPointCloud.from_geometry(
        depth=clip["depths"][frames[0]],
        intrinsics=clip["intrinsics"][frames[0]],
        cam2world=clip["poses_c2w"][frames[0]],
        latent=latents[0].unsqueeze(1),
        mask=None,
        device=device,
    )
    seed_valid = lpc.valid_mask.bool().clone()
    if int(seed_valid.sum()) == 0:
        raise ValueError(f"{src_dir.name}: no valid points in the seed frame")

    points, feats, valid, frame_of = accumulate(
        clip, lpc, frames, write_t, LatentPointCloud, latents, device=device)

    _require_finite("points_world", points[valid], clip["clip_id"])

    write_sets = rollout_write_sets(n_real, rungs)
    rung_masks = []
    for ws in write_sets:
        in_set = torch.zeros_like(valid)
        for i in ws:
            in_set |= (frame_of == i)
        rung_masks.append(valid & in_set)

    # match rollout bounds: start tight, then add 1.5x headroom when points leave the box
    time_bounds = rollout_time_bounds(num_time_steps, n_real)
    bounds_per_rung = [
        AdaptiveBounds.from_steps(
            [0], points.unsqueeze(0), rung_masks[0].unsqueeze(0),
            margin_frac=0.02, fixed_time_bounds=time_bounds)]
    for r in range(1, len(write_sets)):
        new_times = sorted(set(write_sets[r]) - set(write_sets[r - 1]))
        new_mask = valid & torch.isin(
            frame_of, torch.tensor(new_times, dtype=frame_of.dtype))
        if not bool(new_mask.any()):
            bounds_per_rung.append(bounds_per_rung[-1])
            continue
        inc_lo, inc_hi = incoming_xyz_bounds_from_steps(
            [0], points.unsqueeze(0), new_mask.unsqueeze(0), margin_frac=0.02)
        prev = bounds_per_rung[-1]
        if incoming_xyz_inside_bounds(prev, inc_lo, inc_hi):
            bounds_per_rung.append(prev)
        else:
            bounds_per_rung.append(make_reserved_xyz_fixedt_bounds(
                prev, inc_lo, inc_hi, num_time_steps))

    for b in bounds_per_rung:
        _require_finite("bounds.lo", b.lo, clip["clip_id"])
        _require_finite("bounds.hi", b.hi, clip["clip_id"])
        if bool(((b.hi[:3] - b.lo[:3]) <= 0).any()):
            raise ValueError(f"{clip['clip_id']}: degenerate spatial bounds")

    seed_feats = feats[:n_cells][seed_valid]
    mean = seed_feats.mean(0)
    std = seed_feats.std(0).clamp_min(1e-6)
    F = (feats[valid] - mean) / std

    dirs_write = write_dirs(clip, frames, lat_h, lat_w)
    fo_valid = frame_of[valid]
    cell_of = torch.empty_like(fo_valid)
    seed_cells = torch.nonzero(seed_valid, as_tuple=False).squeeze(1)
    cell_of[:seed_cells.numel()] = seed_cells
    cursor = seed_cells.numel()
    for i, f in enumerate(frames):
        if i == write_t:
            continue
        n = int((fo_valid == i).sum())
        if n == 0:
            continue
        fl_valid = _frame_valid_cells(clip, f, lat_h, lat_w, device)
        if fl_valid.numel() != n:
            raise ValueError(
                f"frame {f}: {fl_valid.numel()} valid cells but {n} points")
        cell_of[cursor:cursor + n] = fl_valid
        cursor += n
    if cursor != fo_valid.numel():
        raise ValueError(f"cell bookkeeping covered {cursor} of {fo_valid.numel()}")
    VD = dirs_write[fo_valid, cell_of]

    centres = torch.stack([
        torch.as_tensor(clip["poses_c2w"][f], dtype=torch.float32)[:3, 3]
        for f in frames])
    origins_frame = torch.stack([
        torch.stack([normalize_origin(c, b).reshape(3) for c in centres])
        for b in bounds_per_rung])
    dirs_read, _ = build_readout_rays(clip, lpc, bounds_per_rung[0], frames=frames)

    lpc.points_world = points
    sels, hits = [], []
    for m in rung_masks:
        lpc.valid_mask = m
        s_k, h_k = [], []
        for idx in frames:
            K_lat = scale_intrinsics_to_latent(
                clip["intrinsics"][idx], lat_h, lat_w, *clip["depths"].shape[1:3])
            s, h = indexed_project(lpc, clip["poses_c2w"][idx], K_lat)
            s_k.append(s)
            h_k.append(h)
        sels.append(torch.stack(s_k))
        hits.append(torch.stack(h_k))

    return {
        "clip_id": src_dir.name,
        "num_real_times": n_real,
        "num_time_steps": int(num_time_steps),
        "frames": frames,
        "write_t": write_t,
        "latent_hw": (lat_h, lat_w),
        "n_cells": n_cells,
        "bounds_lo": torch.stack([b.lo for b in bounds_per_rung]),
        "bounds_hi": torch.stack([b.hi for b in bounds_per_rung]),
        "mean": mean, "std": std,
        "points_world": points,
        "valid": valid,
        "frame_of": frame_of.to(torch.int16),
        "F": F.half(),
        "VD": VD.half(),
        "origins_frame": origins_frame,
        "dirs": dirs_read.half(),
        "write_sets": write_sets,
        "sel": torch.stack(sels).to(torch.int32),
        "hit": torch.stack(hits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]),
                    help="repository root containing src/lsm (default: this checkout)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    ap.add_argument("--num-time-steps", type=int, default=DEFAULT_NUM_TIME_STEPS)
    ap.add_argument("--ids", default=None,
                    help="JSON list / {'ids': [...]} / one-per-line file")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if os.environ.get("OMP_NUM_THREADS") != "1":
        print("WARNING: OMP_NUM_THREADS != 1; the projection scatter is "
              "nondeterministic above one thread", file=sys.stderr)
    torch.set_num_threads(1)

    sys.path.insert(0, str(Path(a.repo_root) / "src"))
    from lsm.latent_point_cloud import LatentPointCloud

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(d for d in src.iterdir() if d.is_dir() and d.name.isdigit())
    if a.ids:
        p = Path(a.ids)
        raw = json.loads(p.read_text()) if p.suffix == ".json" else p.read_text().split()
        want = set(raw["ids"] if isinstance(raw, dict) else raw)
        clips = [c for c in clips if c.name in want]
    if a.limit:
        clips = clips[:a.limit]
    clips = clips[a.shard::a.n_shards]

    missing = check_encoder_complete(clips)
    if missing:
        print(f"FATAL: {len(missing)} clips lack clip.pt; run "
              f"`run_video_vae_encode --video-keys clip` first", file=sys.stderr)
        return 1

    t0, failures = time.time(), []
    for i, c in enumerate(clips, 1):
        dst = out / f"{c.name}.pt"
        if dst.exists():
            continue
        try:
            pack = build_rollout_pack(
                c, LatentPointCloud, num_frames=a.num_frames,
                num_time_steps=a.num_time_steps)
            torch.save(pack, dst)
        except Exception as exc:
            failures.append({"clip": c.name, "err": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        if i % 25 == 0:
            print(f"[shard {a.shard}] {i}/{len(clips)} "
                  f"({time.time() - t0:.0f}s, {len(failures)} failed)", flush=True)

    print(json.dumps({"shard": a.shard, "clips": len(clips),
                      "wall_s": round(time.time() - t0, 1),
                      "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
