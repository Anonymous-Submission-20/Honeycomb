from typing import Any

import torch

DEFAULT_LATENT_CHANNELS = 48


def cells_to_vace_latents(cell_tokens: torch.Tensor, lat_h: int,
                          lat_w: int) -> torch.Tensor:
    # reshape cell tokens [frames, height * width, 48] into [1, 48, frames, height, width]
    if cell_tokens.ndim != 3:
        raise ValueError(f"cell_tokens must be [F, N, C], got {tuple(cell_tokens.shape)}")
    frames, n_cells, channels = cell_tokens.shape
    if n_cells != int(lat_h) * int(lat_w):
        raise ValueError(f"N={n_cells} does not match lat_h*lat_w={int(lat_h) * int(lat_w)}")
    if channels != 48:
        raise ValueError(f"expected 48 latent channels, got {channels}")
    return cell_tokens.view(frames, lat_h, lat_w, channels).permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def cells_to_vace_mask(cell_mask: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
    if cell_mask.ndim != 2:
        raise ValueError(f"cell_mask must be [F, N], got {tuple(cell_mask.shape)}")
    frames, n_cells = cell_mask.shape
    if n_cells != int(lat_h) * int(lat_w):
        raise ValueError(f"N={n_cells} does not match lat_h*lat_w={int(lat_h) * int(lat_w)}")
    return cell_mask.view(frames, lat_h, lat_w).unsqueeze(0).unsqueeze(0).contiguous()


def validate_memory_latents(memory_latents: torch.Tensor,
                            latent_channels: int = DEFAULT_LATENT_CHANNELS,
                            name: str = "memory_latents") -> torch.Tensor:
    if not torch.is_tensor(memory_latents):
        raise TypeError(f"{name} must be a torch.Tensor")
    if memory_latents.ndim != 5:
        raise ValueError(
            f"{name} must have shape [B, C, F, H, W], got {tuple(memory_latents.shape)}")
    if int(memory_latents.shape[1]) != int(latent_channels):
        raise ValueError(
            f"{name} must have {latent_channels} channels, got {memory_latents.shape[1]}")
    if any(int(x) <= 0 for x in memory_latents.shape):
        raise ValueError(f"{name} dimensions must all be positive, got {tuple(memory_latents.shape)}")
    return memory_latents


def _channel_stat_view(stat: Any, like: torch.Tensor, name: str) -> torch.Tensor:
    stat_t = torch.as_tensor(stat, dtype=like.dtype, device=like.device)
    if stat_t.ndim == 1:
        if stat_t.numel() != like.shape[1]:
            raise ValueError(f"{name} length must match latent channels {like.shape[1]}")
        return stat_t.view(1, like.shape[1], 1, 1, 1)
    return stat_t


def unnormalize_latents(memory_norm: torch.Tensor, mean: Any, std: Any) -> torch.Tensor:
    validate_memory_latents(memory_norm, latent_channels=memory_norm.shape[1],
                            name="memory_norm")
    mean_t = _channel_stat_view(mean, memory_norm, "mean")
    std_t = _channel_stat_view(std, memory_norm, "std")
    return memory_norm * std_t + mean_t
