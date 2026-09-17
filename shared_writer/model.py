import torch
import torch.nn as nn
import torch.nn.functional as Fn

from shared_writer.splat import bilinear_splat

SPECS = [(("x", "y"), "z"), (("x", "z"), "y"), (("y", "z"), "x")]

R_CHANNELS = (48, 48, 48)

FEATURE_DIM = sum(R_CHANNELS)


def plane_resolutions(extents, max_res=256):
    # give the longest axis max_res cells and scale the others by their extents
    ex, ey, ez = [float(e) for e in extents]
    m = max(ex, ey, ez)
    return {a: max(32, int(round(max_res * e / m)))
            for a, e in zip("xyz", (ex, ey, ez))}


class _ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class _UNet(nn.Module):
    # depth 0 uses a single 1x1 convolution

    def __init__(self, c_in, base, depth):
        super().__init__()
        self.depth = depth
        if depth == 0:
            self.linear = nn.Conv2d(c_in, base, 1)
            return
        self.enc = nn.ModuleList()
        ch = c_in
        chans = [base * (2 ** i) for i in range(depth + 1)]
        for c in chans:
            self.enc.append(_ConvBlock(ch, c))
            ch = c
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.dec.append(_ConvBlock(chans[i + 1] + chans[i], chans[i]))

    def forward(self, x):
        if self.depth == 0:
            return self.linear(x)
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = Fn.max_pool2d(x, 2, ceil_mode=True)
        for block in self.dec:
            skip = skips.pop()
            x = Fn.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                               align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return x


M512 = {
    "base": 48,
    "depth": 0,
    "point_dim": 768,
    "point_layers": 3,
    "contrib_dim": 128,
    "reader_hidden": 768,
    "max_res": 512,
}


class PlaneWriter(nn.Module):
    def __init__(self, feat_dim=48, base=M512["base"], refine_depth=M512["depth"],
                 hidden=M512["reader_hidden"], max_res=M512["max_res"],
                 point_dim=M512["point_dim"], contrib_dim=M512["contrib_dim"],
                 p_in_encoder=False, point_layers=M512["point_layers"],
                 ranks=R_CHANNELS):
        super().__init__()
        self.max_res = max_res
        self.ranks = tuple(int(r) for r in ranks)
        if len(self.ranks) != len(SPECS):
            raise ValueError(
                f"ranks must have {len(SPECS)} entries, got {self.ranks}")
        self.p_in_encoder = bool(p_in_encoder)
        in_pt = feat_dim + 3 + 3 + 1 + (3 if self.p_in_encoder else 0)
        layers, ch = [], in_pt
        for _ in range(max(1, int(point_layers))):
            layers += [nn.Linear(ch, point_dim), nn.ReLU(inplace=True)]
            ch = point_dim
        self.point_mlp = nn.Sequential(*layers)
        self.contrib_spatial = nn.ModuleList(
            [nn.Linear(point_dim, contrib_dim) for _ in range(3)])
        self.contrib_st = nn.ModuleList(
            [nn.Linear(point_dim, contrib_dim) for _ in range(3)])
        in_ch = contrib_dim + 1               # include log density so the refiner can distinguish sparse cells
        self.spatial_trunk = _UNet(in_ch, base, refine_depth)
        self.spatial_heads = nn.ModuleList(
            [nn.Conv2d(base, r, 1) for r in self.ranks])
        self.st_trunk = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.st_heads = nn.ModuleList(
            [nn.Conv2d(base, r, 1) for r in self.ranks])
        for head in self.st_heads:            # start spatiotemporal planes at 1 so multiplication preserves spatial features
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.reader = nn.Sequential(
            nn.Linear(sum(self.ranks) + 6, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim))
        nn.init.constant_(self.reader[-1].bias, 0.0)

    def write(self, p_norm, t_norm, feats, viewdirs, origins, extents, t_res,
              res=None, return_density=False):
        res = plane_resolutions(extents, self.max_res) if res is None else res
        parts = [feats, viewdirs, origins, t_norm.reshape(-1, 1)]
        if self.p_in_encoder:
            parts.append(p_norm)
        enc = self.point_mlp(torch.cat(parts, dim=1))
        c = {"x": p_norm[:, 0], "y": p_norm[:, 1], "z": p_norm[:, 2]}
        spatial, st, d_spatial, d_st = [], [], [], []
        for i, ((aw, ah), ac) in enumerate(SPECS):
            uv = torch.stack([c[aw], c[ah]], dim=1)
            m = bilinear_splat(uv, self.contrib_spatial[i](enc),
                               res[ah], res[aw])
            d_spatial.append(m[-1:].unsqueeze(0))
            m = torch.cat([m[:-1], torch.log1p(m[-1:])], dim=0)
            x = self.spatial_trunk(m.unsqueeze(0))
            spatial.append(self.spatial_heads[i](x))

            ut = torch.stack([c[ac], t_norm], dim=1)
            m = bilinear_splat(ut, self.contrib_st[i](enc), t_res, res[ac])
            d_st.append(m[-1:].unsqueeze(0))
            m = torch.cat([m[:-1], torch.log1p(m[-1:])], dim=0)
            y = self.st_trunk(m.unsqueeze(0))
            st.append(1.0 + self.st_heads[i](y))
        if return_density:
            return spatial, st, d_spatial, d_st
        return spatial, st


def _sample(plane, u, v):
    grid = torch.stack([u, v], -1).view(1, 1, -1, 2)
    return Fn.grid_sample(plane, grid, mode="bilinear", padding_mode="border",
                          align_corners=True).view(plane.shape[1], -1)


def functional_plane_features(spatial, st, p_norm, t_norm):
    c = {"x": p_norm[:, 0], "y": p_norm[:, 1], "z": p_norm[:, 2]}
    feats = []
    for i, ((aw, ah), ac) in enumerate(SPECS):
        fs = _sample(spatial[i], c[aw], c[ah])
        ft = _sample(st[i], c[ac], t_norm)
        feats.append(fs * ft)
    return torch.cat(feats, 0).T


def functional_forward_ray(spatial, st, reader, p_norm, viewdir, origin,
                           t_norm):
    x = torch.cat([functional_plane_features(spatial, st, p_norm, t_norm),
                   viewdir, origin], dim=1)
    return reader(x)
