import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter.corpus_utils import (  # noqa: E402
    build_write_inputs, build_point_cloud, build_readout_rays, indexed_project,
    load_clip, scale_intrinsics_to_latent,
)
from honeycomb import (  # noqa: E402
    AdaptiveBounds, query_projected_adaptive_readout,
)
from shared_writer.field import load_writer, write_field  # noqa: E402
from recurrent_writer.incremental import load_incremental_writer  # noqa: E402
from replacement_writer.prep_pack_rollout import DEFAULT_NUM_TIME_STEPS  # noqa: E402
from replacement_writer.rollout_timeline import rollout_time_bounds  # noqa: E402
from recurrent_writer.frame import IncrementalFrame  # noqa: E402
from recurrent_writer.incremental import StateField  # noqa: E402
from recurrent_writer.time_map import make_time_map  # noqa: E402

REGENERATED = ("train_preceding_scene_proj_rgb.pt", "train_target_scene_proj_rgb.pt")


def _denorm_xyz(norm, bounds):
    lo = bounds.lo[:3].to(norm.device, norm.dtype)
    hi = bounds.hi[:3].to(norm.device, norm.dtype)
    return 0.5 * (norm + 1.0) * (hi - lo) + lo


def build_one(clip_dir, out_dir, LatentPointCloud, save_projection_latent,
              device, writer):
    t_start = time.time()
    clip = load_clip(clip_dir)
    frames = clip["proj_P_idx"] + clip["proj_T_idx"]
    n_p, n_t = len(clip["proj_P_idx"]), len(clip["proj_T_idx"])
    K = len(frames)
    if clip["scene_idx"] not in frames:
        raise ValueError(f"scene_idx {clip['scene_idx']} not among projection frames")
    write_t = frames.index(clip["scene_idx"])

    lpc = build_point_cloud(clip, LatentPointCloud, device="cpu")
    lat_h, lat_w = lpc.latent_hw
    n_valid = int(lpc.valid_mask.sum())
    if n_valid == 0:
        raise ValueError("no valid points")

    # use the rollout time grid the writer was trained on, starting at time 0
    t_res = DEFAULT_NUM_TIME_STEPS
    time_bounds = rollout_time_bounds(t_res)
    write_tau = 0.0
    bounds = AdaptiveBounds.from_steps(
        [0], lpc.points_world.unsqueeze(0), lpc.valid_mask.unsqueeze(0),
        margin_frac=0.02, fixed_time_bounds=time_bounds)
    data, mean, std = build_write_inputs(clip, lpc, bounds, write_tau)
    if hasattr(writer, "update"):
        P, VD, ORG, F, Tau = data
        frame = IncrementalFrame(
            bounds, make_time_map(getattr(writer, "ckpt_meta", {}).get("time_map")
                                  or "affine",
                                  chunk_span=float(
                                      getattr(writer, "ckpt_meta", {}).get("chunk_span")
                                      or 8.0)))
        state = writer.update(
            None,
            {"points_world": P.to(device), "times": Tau.to(device),
             "feats": F.to(device), "viewdirs": VD.to(device),
             # convert normalized origins back to world coordinates for the writer
             "origins_world": _denorm_xyz(ORG.to(device), bounds)},
            frame=frame, t_res=t_res)
        model = StateField(state, writer.writer.reader)
    else:
        model = write_field(writer, data, bounds, t_res, device)

    # project on one CPU thread so overlapping points resolve consistently
    sel, hit = [], []
    for idx in frames:
        K_lat = scale_intrinsics_to_latent(clip["intrinsics"][idx], lat_h, lat_w,
                                           *clip["depths"].shape[1:3])
        s, h = indexed_project(lpc, clip["poses_c2w"][idx], K_lat)
        sel.append(s)
        hit.append(h)
    sel, hit = torch.stack(sel), torch.stack(hit)
    dirs, origins = build_readout_rays(clip, lpc, bounds, frames=frames)

    out = query_projected_adaptive_readout(
        model, lpc.points_world, sel, hit, list(range(K)), dirs, origins,
        mean, std, device, lat_h=lat_h, lat_w=lat_w,
        source_steps=torch.full((lpc.points_world.shape[0],), write_tau))
    hexp = out["memory_latents_raw"][0].permute(1, 0, 2, 3)

    # use the same serializer as the LMDB packer
    out_dir.mkdir(parents=True, exist_ok=True)
    save_projection_latent(hexp[:n_p].numpy(), out_dir / REGENERATED[0])
    save_projection_latent(hexp[n_p:].numpy(), out_dir / REGENERATED[1])

    row = {"clip_id": clip["clip_id"], "n_valid_points": n_valid,
           "n_preceding": n_p, "n_target": n_t, "write_t": write_t,
           "write_tau": write_tau, "t_res": t_res,
           "memory_producer": "writer",
           "hit_frac": float(hit.float().mean()),
           "wall_s": round(time.time() - t_start, 1)}
    return row


def mirror_rest(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(os.listdir(src_dir)):
        if entry in REGENERATED:
            continue
        dst = out_dir / entry
        if dst.is_symlink() or dst.exists():
            continue
        src = src_dir / entry
        dst.symlink_to(os.path.realpath(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source train/ dir of clip folders")
    ap.add_argument("--out", required=True, help="destination train/ dir")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]),
                    help="repository root containing src/lsm (default: this checkout)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--writer-ckpt", required=True,
                    help="writer checkpoint (replacement or incremental); "
                         "architecture and max_res are taken from the "
                         "checkpoint's own recorded args.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    # keep exclusions separate for each corpus because clip IDs can overlap
    ap.add_argument("--exclude", default=None,
                    help="clip ids to skip: a file (JSON list / {'ids': []} / "
                         "one per line) or a literal comma-separated list")
    ap.add_argument("--force", action="store_true", help="rebuild existing outputs")
    a = ap.parse_args()

    # one CPU thread keeps the projection scatter deterministic
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(a.repo_root) / "src"))
    sys.path.insert(0, str(Path(a.repo_root)))
    from lsm.latent_point_cloud import LatentPointCloud
    from data_process.dataset_writer import save_projection_latent

    device = torch.device(a.device)
    src_root, out_root = Path(a.src), Path(a.out)

    try:
        writer = load_incremental_writer(a.writer_ckpt, device=device)
    except (ValueError, KeyError):
        writer = load_writer(a.writer_ckpt, device=device)

    clips = sorted(d for d in os.listdir(src_root) if (src_root / d).is_dir())
    if a.only:
        want = set(a.only.split(","))
        clips = [c for c in clips if c in want]

    # exclude clips before sharding so all workers use the same clip list
    excluded_ids = []
    if a.exclude:
        p = Path(a.exclude)
        if p.exists():
            raw = p.read_text()
            try:
                obj = json.loads(raw)
                want_out = obj["ids"] if isinstance(obj, dict) else obj
            except json.JSONDecodeError:
                want_out = raw.split()
        else:
            want_out = a.exclude.split(",")
        want_out = {str(x).strip() for x in want_out if str(x).strip()}
        excluded_ids = sorted(want_out & set(clips))
        missing = sorted(want_out - set(clips))
        clips = [c for c in clips if c not in want_out]
        print(f"excluded {len(excluded_ids)} clip(s): {excluded_ids}"
              + (f"  (not present in this corpus, ignored: {missing})"
                 if missing else ""), flush=True)

    clips = clips[a.shard::a.n_shards]
    if a.limit:
        clips = clips[:a.limit]

    producer = (f"writer {Path(a.writer_ckpt).name} "
                f"(max_res={writer.max_res} step={writer.ckpt_meta.get('step')})")
    print(f"shard {a.shard}/{a.n_shards}: {len(clips)} clips  device={a.device}\n"
          f"  memory producer: {producer}\n"
          f"  (projector: CPU, 1 thread)", flush=True)

    rows, failures, skipped = [], [], 0
    t0 = time.time()
    for i, cid in enumerate(clips):
        out_dir = out_root / cid
        done = all((out_dir / f).exists() for f in REGENERATED)
        if done and not a.force:
            skipped += 1
            continue
        try:
            row = build_one(src_root / cid, out_dir, LatentPointCloud,
                            save_projection_latent, device, writer)
            mirror_rest(src_root / cid, out_dir)
            rows.append(row)
            print(f"[{i+1}/{len(clips)}] {cid}  N={row['n_valid_points']:5d}  "
                  f"{row['wall_s']:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"clip_id": cid, "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})
            print(f"[{i+1}/{len(clips)}] {cid}  FAILED  {type(exc).__name__}: {exc}",
                  flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    # record which writer produced the corpus
    report = {"shard": a.shard, "n_shards": a.n_shards, "built": len(rows),
              "skipped_existing": skipped, "failed": len(failures),
              "excluded": len(excluded_ids), "excluded_ids": excluded_ids,
              "device": a.device,
              "memory_producer": "writer",
              "writer_ckpt": a.writer_ckpt,
              "writer_meta": writer.ckpt_meta,
              "max_res": writer.max_res,
              "wall_s": round(time.time() - t0, 1), "rows": rows,
              "failures": failures}
    rp = out_root / f"_build_report_shard{a.shard}of{a.n_shards}.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nbuilt {len(rows)}, skipped {skipped}, failed {len(failures)} "
          f"in {report['wall_s']}s -> {rp}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
