import torch
import torch.nn as nn

from shared_writer.model import PlaneWriter, functional_forward_ray


class WrittenField(nn.Module):
    def __init__(self, spatial, st, reader, bounds):
        super().__init__()
        self.spatial = list(spatial)
        self.st = list(st)
        self.reader = reader
        self.bounds = bounds

    def forward_ray(self, p_world, viewdir, origin, tau):
        p_norm, t_norm = self.bounds.world_to_norm(p_world, tau, clamp=True)
        return functional_forward_ray(
            self.spatial, self.st, self.reader,
            p_norm.to(p_world.device), viewdir, origin, t_norm.to(p_world.device))

    @torch.no_grad()
    def warp_to_bounds(self, new_bounds, min_growth_frac=1e-4, force=False):
        # only update the bounds here; the caller rebuilds the planes from the stored points
        old = self.bounds
        changed = bool(torch.any(torch.abs(old.lo - new_bounds.lo) > 1e-8).item()
                       or torch.any(torch.abs(old.hi - new_bounds.hi) > 1e-8).item())
        self.bounds = new_bounds
        return {"old_bounds": old.to_json(), "new_bounds": new_bounds.to_json(),
                "warped": changed, "force": bool(force),
                "regenerated_by_writer": True}


def load_writer(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    writer = PlaneWriter(
        base=a["base"], refine_depth=a["depth"], point_dim=a["point_dim"],
        contrib_dim=a["contrib_dim"], hidden=a["reader_hidden"],
        p_in_encoder=a.get("p_input", False),
        point_layers=a["point_layers"], max_res=a["max_res"],
        ranks=tuple(a["ranks"]))
    writer.load_state_dict(ckpt["model"])
    writer.eval().to(device)
    for p in writer.parameters():
        p.requires_grad_(False)
    writer.ckpt_meta = {"path": str(ckpt_path),
                        "epoch": ckpt.get("epoch"), "step": ckpt.get("step")}
    return writer


@torch.no_grad()
def write_field(writer, data, bounds, num_time_steps, device):
    P, VD, ORG, F, Tau = [d.to(device) for d in data]
    p_norm, t_norm = bounds.world_to_norm(P, Tau, clamp=True)
    extents = bounds.span()[:3].tolist()
    spatial, st = writer.write(
        p_norm.to(device), t_norm.to(device), F, VD, ORG,
        extents, int(num_time_steps))
    return WrittenField(spatial, st, writer.reader, bounds)
