import torch

from .latents import cells_to_vace_latents, cells_to_vace_mask, unnormalize_latents


@torch.no_grad()
def query_projected_adaptive_readout(
    model,
    memory_points: torch.Tensor,
    selected_indices: torch.Tensor,
    projection_hit_mask: torch.Tensor,
    target_steps,
    dirs: torch.Tensor,
    origins: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device,
    lat_h: int,
    lat_w: int,
    query_batch: int = 65536,
    source_steps=None,
):
    device = torch.device(device)
    n_frames, n_cells = selected_indices.shape
    pred_norm = torch.zeros(n_frames, n_cells, 48, dtype=torch.float32)
    hit_mask = projection_hit_mask.detach().cpu().bool().clone()
    memory_points = memory_points.to(device=device, dtype=torch.float32)
    if source_steps is not None:
        source_steps = torch.as_tensor(source_steps).reshape(-1).to(
            device=device, dtype=torch.float32)
        if int(source_steps.numel()) != int(memory_points.shape[0]):
            raise ValueError(
                f"source_steps has {int(source_steps.numel())} entries but there are "
                f"{int(memory_points.shape[0])} memory points; it must carry the write "
                "step of every point (build_memory_geometry returns exactly this)")

    model.eval()
    for frame_i, k in enumerate(target_steps):
        cells = torch.nonzero(hit_mask[frame_i], as_tuple=False).squeeze(1)
        if cells.numel() == 0:
            continue
        mem_idx = selected_indices[frame_i, cells].to(device)
        for start in range(0, int(cells.numel()), int(query_batch)):
            end = min(start + int(query_batch), int(cells.numel()))
            c_cpu = cells[start:end]
            idx = mem_idx[start:end]
            p = memory_points[idx]
            viewdir = dirs[int(k), c_cpu].to(device)
            origin = origins[int(k), c_cpu].to(device)
            if source_steps is None:
                tau = torch.full((end - start,), float(k), dtype=torch.float32, device=device)
            else:
                tau = source_steps[idx]          # query each point at the time it was written
            pred = model.forward_ray(p, viewdir, origin, tau)
            pred_norm[frame_i, c_cpu] = pred.detach().cpu()

    pred_norm = pred_norm * hit_mask.unsqueeze(-1).to(pred_norm.dtype)
    memory_latents_norm = cells_to_vace_latents(pred_norm, lat_h=lat_h, lat_w=lat_w)
    memory_mask = cells_to_vace_mask(hit_mask, lat_h=lat_h, lat_w=lat_w).bool().contiguous()
    memory_latents_raw = unnormalize_latents(memory_latents_norm, mean, std)
    memory_latents_raw = memory_latents_raw * memory_mask.to(memory_latents_raw.dtype)
    return {
        "memory_latents_raw": memory_latents_raw.contiguous(),
        "memory_mask": memory_mask,
        "projection_hit_mask": projection_hit_mask.cpu().bool(),
    }

def masked_raw_mse(memory_latents_raw: torch.Tensor, target_latents_raw: torch.Tensor,
                   memory_mask: torch.Tensor):
    if memory_latents_raw.shape != target_latents_raw.shape:
        raise ValueError("memory_latents_raw and target_latents_raw must have matching shapes")
    err = ((memory_latents_raw - target_latents_raw) ** 2).mean(dim=1, keepdim=True)
    mask = memory_mask.to(dtype=torch.bool)
    if not mask.any():
        return {"raw_mse_hit_only": None, "raw_mse_all": float(err.mean())}
    return {
        "raw_mse_hit_only": float(err[mask].mean()),
        "raw_mse_all": float(err.mean()),
    }
