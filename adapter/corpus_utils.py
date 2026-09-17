import json
from pathlib import Path

import numpy as np
import torch


def load_clip(clip_dir):
    clip_dir = Path(clip_dir)
    g = np.load(clip_dir / "geometry.npz")
    meta = json.loads((clip_dir / "train_sample.json").read_text())
    proj = torch.load(clip_dir / "train_target_scene_proj_rgb.pt",
                      map_location="cpu", weights_only=False)["latent"].float()
    return {
        "clip_id": clip_dir.name,
        "depths": g["depths"],
        "poses_c2w": g["poses_c2w"],
        "intrinsics": g["intrinsics"],
        "scene_idx": int(meta["scene_idx"]),
        "proj_T_idx": [int(i) for i in meta["proj_T_idx"]],
        "proj_P_idx": [int(i) for i in meta.get("proj_P_idx", [])],
        "target_scene_proj": proj,
    }


def scale_intrinsics_to_latent(intr, lat_h, lat_w, img_h, img_w):
    sx, sy = lat_w / img_w, lat_h / img_h
    K = torch.as_tensor(intr, dtype=torch.float32).clone()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def build_point_cloud(clip, LatentPointCloud, device="cpu"):
    lat_f, lat_c, lat_h, lat_w = clip["target_scene_proj"].shape
    scene_idx = clip["scene_idx"]
    if clip["proj_T_idx"][0] != scene_idx:
        raise ValueError(
            f"feature recovery assumes proj_T_idx[0] == scene_idx, got "
            f"{clip['proj_T_idx'][0]} vs {scene_idx}")

    # these latent features are replaced with the packed features below
    placeholder = torch.zeros(lat_c, 1, lat_h, lat_w, dtype=torch.float32)
    lpc = LatentPointCloud.from_geometry(
        depth=clip["depths"][scene_idx],
        intrinsics=clip["intrinsics"][scene_idx],
        cam2world=clip["poses_c2w"][scene_idx],
        latent=placeholder,
        mask=None,
        device=device,
    )

    scene_proj = clip["target_scene_proj"][0].to(device)
    features = scene_proj.permute(1, 2, 0).reshape(-1, lat_c)    # flatten cells in row-major order: cell = v * width + u
    hit = (scene_proj.abs().sum(0) > 0).reshape(-1)

    lpc.features = features
    # keep only points that also appear in the source-view projection
    lpc.valid_mask = lpc.valid_mask.to(device) & hit
    return lpc


def indexed_project(lpc, cam2world, intrinsics):
    # use the same z-buffer rule as LatentPointCloud.project, but return point indices
    device = lpc.points_world.device
    cam2world = torch.as_tensor(cam2world, dtype=torch.float32, device=device)
    K = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    h, w = lpc.latent_hw
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    world2cam = torch.inverse(cam2world)
    pts_cam = (lpc.points_world @ world2cam[:3, :3].T) + world2cam[:3, 3]
    z = pts_cam[:, 2]
    u_int = torch.round((pts_cam[:, 0] * fx / z) + cx).long()
    v_int = torch.round((pts_cam[:, 1] * fy / z) + cy).long()

    in_bounds = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    proj_valid = lpc.valid_mask & in_bounds & (z > 0)

    idx_all = torch.arange(lpc.points_world.shape[0], device=device)
    order = torch.argsort(z[proj_valid], descending=True)
    flat = (v_int[proj_valid][order] * w + u_int[proj_valid][order])

    selected = torch.full((h * w,), -1, dtype=torch.long, device=device)
    selected[flat] = idx_all[proj_valid][order]
    hit = selected >= 0
    return selected.clamp_min(0), hit


def cell_rays(cam2world, K, lat_h, lat_w):
    # use integer cell coordinates to match point-cloud projection
    c2w = torch.as_tensor(cam2world, dtype=torch.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = torch.meshgrid(torch.arange(lat_h, dtype=torch.float32),
                          torch.arange(lat_w, dtype=torch.float32), indexing="ij")
    ray_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    ray_world = ray_cam.reshape(-1, 3) @ c2w[:3, :3].T
    ray_world = ray_world / ray_world.norm(dim=-1, keepdim=True)
    return ray_world, c2w[:3, 3]


def build_write_inputs(clip, lpc, bounds, write_t):
    lat_h, lat_w = lpc.latent_hw
    scene_idx = clip["scene_idx"]
    img_h, img_w = clip["depths"].shape[1:3]
    K = scale_intrinsics_to_latent(clip["intrinsics"][scene_idx], lat_h, lat_w, img_h, img_w)

    valid = lpc.valid_mask
    P = lpc.points_world[valid].float()
    feats = lpc.features[valid].float()

    dirs_all, origin = cell_rays(clip["poses_c2w"][scene_idx], K, lat_h, lat_w)
    VD = dirs_all[valid]
    ORG = normalize_origin(origin, bounds).expand(P.shape[0], 3).contiguous()

    mean = feats.mean(0)
    std = feats.std(0).clamp_min(1e-6)
    F = (feats - mean) / std
    Tau = torch.full((P.shape[0],), float(write_t))
    return (P, VD, ORG, F, Tau), mean, std


def normalize_origin(origin, bounds):
    lo, hi = bounds.lo[:3], bounds.hi[:3]
    return (2.0 * (origin.reshape(1, 3) - lo) / (hi - lo).clamp_min(1e-8) - 1.0)


def build_readout_rays(clip, lpc, bounds, frames=None):
    lat_h, lat_w = lpc.latent_hw
    img_h, img_w = clip["depths"].shape[1:3]
    frames = clip["proj_T_idx"] if frames is None else frames
    dirs, origins = [], []
    for idx in frames:
        K = scale_intrinsics_to_latent(clip["intrinsics"][idx], lat_h, lat_w, img_h, img_w)
        d, o = cell_rays(clip["poses_c2w"][idx], K, lat_h, lat_w)
        dirs.append(d)
        origins.append(normalize_origin(o, bounds).expand(lat_h * lat_w, 3))
    return torch.stack(dirs), torch.stack(origins).contiguous()
