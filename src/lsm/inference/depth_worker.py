import sys
import numpy as np
import torch
import torch.nn.functional as F
from vipe.priors.depth.dav3.api import DepthAnything3


def main():
    with np.load(sys.argv[1]) as inputs:
        images = inputs["images"]
        focal_lengths = inputs["fx"].astype(np.float64)
    model = (
        DepthAnything3.from_pretrained(
            "depth-anything/DA3METRIC-LARGE", model_name="da3metric-large"
        )
        .cuda()
        .eval()
    )
    depths = []
    with torch.inference_mode():
        for image, focal in zip(images, focal_lengths):
            height, width = image.shape[:2]
            result = model.inference(
                [image], process_res_method="upper_bound_resize", process_res=504
            )
            focal_504 = focal / max(width, height) * 504.0
            depth = np.asarray(result.depth).squeeze().astype(np.float32) * np.float32(
                focal_504 / 300.0
            )
            depth = (
                F.interpolate(
                    torch.from_numpy(depth)[None, None],
                    size=(height, width),
                    mode="nearest",
                )
                .squeeze()
                .numpy()
            )
            depths.append(depth.astype(np.float32))
    np.savez_compressed(sys.argv[2], depths=np.stack(depths))


if __name__ == "__main__":
    main()
