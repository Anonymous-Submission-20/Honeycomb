import torch

from .wan_video_dit import DiTBlock


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 4, 8, 12, 16, 20, 24, 28),
        vace_in_dim=49,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=3072,
        num_heads=24,
        ffn_dim=14336,
        eps=1e-6,
        use_reentrant=False,
    ):
        super().__init__()
        if vace_in_dim != 49:
            raise ValueError("Honeycomb VACE requires 48 latent channels plus visibility (49 total).")
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        self.vace_blocks = torch.nn.ModuleList(
            [
                VaceWanAttentionBlock(
                    has_image_input, dim, num_heads, ffn_dim, eps, block_id=i
                )
                for i in self.vace_layers
            ]
        )

        self.vace_patch_embedding = torch.nn.Conv3d(
            vace_in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.use_reentrant = use_reentrant

    def forward(
        self,
        x,
        vace_context,
        context,
        t_mod,
        freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))], dim=1)
                for u in c
            ]
        )

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.vace_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c,
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=self.use_reentrant,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c,
                    x,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=self.use_reentrant,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        hints = torch.unbind(c)[:-1]
        return hints
