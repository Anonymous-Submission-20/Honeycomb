import torch
from torch import nn

from .wan_video_vace import VaceWanModel


def infer_vace_layers(num_dit_layers: int) -> tuple[int, ...]:
    return tuple(range(0, num_dit_layers, 4))


def build_scratch_vace_from_dit(
    dit: nn.Module,
    *,
    use_reentrant: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> VaceWanModel:
    blocks = dit.blocks
    first_block = blocks[0]
    vace = VaceWanModel(
        vace_layers=infer_vace_layers(len(blocks)),
        patch_size=tuple(int(value) for value in dit.patch_size),
        has_image_input=bool(getattr(dit, "has_image_input", False)),
        dim=int(dit.dim),
        num_heads=int(first_block.num_heads),
        ffn_dim=int(first_block.ffn_dim),
        eps=float(getattr(first_block.norm1, "eps", 1e-6)),
        use_reentrant=use_reentrant,
    )
    allowed = {"before_proj.weight", "before_proj.bias", "after_proj.weight", "after_proj.bias"}
    for vblock, layer in zip(vace.vace_blocks, vace.vace_layers):
        missing, unexpected = vblock.load_state_dict(blocks[layer].state_dict(), strict=False)
        if set(missing) - allowed or unexpected:
            raise RuntimeError(
                f"VACE layer {layer}: missing={missing}, unexpected={unexpected}")
        nn.init.zeros_(vblock.after_proj.weight)
        nn.init.zeros_(vblock.after_proj.bias)
    return vace.to(device=device, dtype=dtype)
