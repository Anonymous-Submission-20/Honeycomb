import torch


def append_visibility_channel(scene: torch.Tensor, *, channel_dim: int) -> torch.Tensor:
    # mark a cell visible when any projected latent channel is nonzero
    visible = scene.ne(0).any(dim=channel_dim, keepdim=True).to(dtype=scene.dtype)
    return torch.cat([scene, visible], dim=channel_dim)
