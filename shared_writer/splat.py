# use the same coordinate convention as grid_sample with align_corners=True

import torch


def _norm_to_pix(u, size):
    return (u.clamp(-1.0, 1.0) + 1.0) * 0.5 * (size - 1)


def bilinear_splat(coords, feats, height, width, eps=1e-8):
    n, c = feats.shape
    device = feats.device
    x = _norm_to_pix(coords[:, 0], width)
    y = _norm_to_pix(coords[:, 1], height)
    x0 = x.floor().long().clamp(0, max(width - 2, 0))
    y0 = y.floor().long().clamp(0, max(height - 2, 0))
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx1 = (x - x0.to(x.dtype)).clamp(0.0, 1.0)
    wy1 = (y - y0.to(y.dtype)).clamp(0.0, 1.0)
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1

    acc = torch.zeros(c + 1, height * width, device=device, dtype=feats.dtype)
    ones = torch.ones(n, 1, device=device, dtype=feats.dtype)
    src = torch.cat([feats, ones], dim=1)  # carry the accumulated weight in an extra channel
    for xi, wx in ((x0, wx0), (x1, wx1)):
        for yi, wy in ((y0, wy0), (y1, wy1)):
            w = (wx * wy).unsqueeze(1)
            acc.index_add_(1, yi * width + xi, (src * w).T.contiguous())
    # normalize out of place so autograd can still use the original values
    weight = acc[c:c + 1]
    normed = acc[:c] / weight.clamp_min(eps)
    return torch.cat([normed, weight], dim=0).reshape(c + 1, height, width)


def splat_planes(p_norm, t_norm, feats, res, t_res, specs):
    c = {"x": p_norm[:, 0], "y": p_norm[:, 1], "z": p_norm[:, 2]}
    spatial_maps, st_maps = [], []
    for (aw, ah), ac in specs:
        uv = torch.stack([c[aw], c[ah]], dim=1)
        spatial_maps.append(bilinear_splat(uv, feats, res[ah], res[aw]))
        ut = torch.stack([c[ac], t_norm], dim=1)
        st_maps.append(bilinear_splat(ut, feats, t_res, res[ac]))
    return spatial_maps, st_maps
