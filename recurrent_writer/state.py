from dataclasses import dataclass
from typing import List

import torch

from .frame import IncrementalFrame
from .warp import PLANE_SPECS, ST_EMPTY, warp_field, warp_weights


@dataclass
class IncrementalState:
    spatial: List[torch.Tensor]      # three planes, each shaped [1, rank, height, width]
    st: List[torch.Tensor]           # three planes, each shaped [1, rank, time, width]
    w_spatial: List[torch.Tensor]    # evidence weights, not point counts
    w_st: List[torch.Tensor]
    frame: IncrementalFrame

    def __post_init__(self):
        n = len(PLANE_SPECS)
        for name, seq in (("spatial", self.spatial), ("st", self.st),
                          ("w_spatial", self.w_spatial), ("w_st", self.w_st)):
            if len(seq) != n:
                raise ValueError(f"{name} must have {n} planes, got {len(seq)}")

    @property
    def plane_shapes(self) -> dict:
        return {"spatial": [tuple(p.shape) for p in self.spatial],
                "st": [tuple(p.shape) for p in self.st]}

    def assert_shapes_match(self, spatial, st) -> None:
        want = self.plane_shapes
        got = {"spatial": [tuple(p.shape) for p in spatial],
               "st": [tuple(p.shape) for p in st]}
        if got != want:
            raise ValueError(
                "candidate planes do not match the memory's pinned shapes; the "
                "splat must target the carried state, not recompute resolutions "
                f"from expanded bounds.\n  expected {want}\n  got      {got}")

    def warped_to(self, new_frame: IncrementalFrame) -> "IncrementalState":
        self.frame.assert_compatible(new_frame)
        tm = self.frame.time_map
        sp, st = warp_field(self.spatial, self.st, self.frame.bounds,
                            new_frame.bounds, tm)
        wsp, wst = warp_weights(self.w_spatial, self.w_st, self.frame.bounds,
                                new_frame.bounds, tm)
        return IncrementalState(sp, st, wsp, wst, new_frame)

    def state_dict(self) -> dict:
        return {
            "spatial": [p.detach().cpu() for p in self.spatial],
            "st": [p.detach().cpu() for p in self.st],
            "w_spatial": [p.detach().cpu() for p in self.w_spatial],
            "w_st": [p.detach().cpu() for p in self.w_st],
            "frame": self.frame.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "IncrementalState":
        for key in ("spatial", "st", "w_spatial", "w_st", "frame"):
            if key not in sd:
                raise ValueError(f"state_dict missing {key!r}")
        return cls(list(sd["spatial"]), list(sd["st"]),
                   list(sd["w_spatial"]), list(sd["w_st"]),
                   IncrementalFrame.from_state_dict(sd["frame"]))

    @classmethod
    def empty_like(cls, ref: "IncrementalState", frame: IncrementalFrame) -> "IncrementalState":
        # empty spatial planes are 0; empty spatiotemporal planes are 1; confidence is 0
        return cls([torch.zeros_like(p) for p in ref.spatial],
                   [torch.full_like(p, ST_EMPTY) for p in ref.st],
                   [torch.zeros_like(p) for p in ref.w_spatial],
                   [torch.zeros_like(p) for p in ref.w_st],
                   frame)
