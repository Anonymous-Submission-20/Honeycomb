from dataclasses import dataclass

import torch

from honeycomb import AdaptiveBounds

from .time_map import TimeMap, make_time_map


@dataclass
class IncrementalFrame:
    bounds: AdaptiveBounds
    time_map: TimeMap

    def to_norm_xyz(self, p_world):
        lo = self.bounds.lo[:3].to(p_world.device, p_world.dtype)
        hi = self.bounds.hi[:3].to(p_world.device, p_world.dtype)
        return 2.0 * (p_world - lo) / (hi - lo).clamp_min(1e-8) - 1.0

    def to_norm_t(self, t_world):
        return self.time_map.to_norm(t_world, self.t_lo, self.t_hi)

    def to_world_t(self, tau):
        return self.time_map.from_norm(tau, self.t_lo, self.t_hi)

    @property
    def t_lo(self) -> float:
        return float(self.bounds.lo[3])

    @property
    def t_hi(self) -> float:
        return float(self.bounds.hi[3])

    def expand_time_by_chunk(self, n_chunks: int = 1) -> "IncrementalFrame":
        # extend the upper time bound by whole chunks and keep the lower bound fixed
        if n_chunks < 1:
            raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
        lo = self.bounds.lo.clone()
        hi = self.bounds.hi.clone()
        hi[3] = hi[3] + float(n_chunks) * self.time_map.chunk_span
        return IncrementalFrame(AdaptiveBounds(lo=lo, hi=hi), self.time_map)

    def assert_compatible(self, other: "IncrementalFrame") -> None:
        if self.time_map.name != other.time_map.name:
            raise ValueError(
                f"time_map family changed mid-memory: {self.time_map.name!r} -> "
                f"{other.time_map.name!r}. Warping assumes one family; a real "
                f"switch needs separate old/new maps.")
        if float(self.time_map.chunk_span) != float(other.time_map.chunk_span):
            raise ValueError(
                f"chunk_span changed mid-memory: {self.time_map.chunk_span} -> "
                f"{other.time_map.chunk_span}; the exponent schedule would jump.")

    def state_dict(self) -> dict:
        return {
            "time_map": self.time_map.name,
            "chunk_span": float(self.time_map.chunk_span),
            "bounds_lo": [float(x) for x in self.bounds.lo],
            "bounds_hi": [float(x) for x in self.bounds.hi],
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "IncrementalFrame":
        for key in ("time_map", "chunk_span", "bounds_lo", "bounds_hi"):
            if key not in sd:
                raise ValueError(f"frame state_dict missing {key!r}")
        return cls(
            AdaptiveBounds(lo=torch.tensor(sd["bounds_lo"], dtype=torch.float32),
                           hi=torch.tensor(sd["bounds_hi"], dtype=torch.float32)),
            make_time_map(sd["time_map"], chunk_span=float(sd["chunk_span"])))


def frames_compatible(a: IncrementalFrame, b: IncrementalFrame) -> bool:
    try:
        a.assert_compatible(b)
    except ValueError:
        return False
    return True
