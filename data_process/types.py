from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SampleIndices:
    t0: int
    preceding_indices: list[int]
    target_indices: list[int]
    candidate_indices: list[int]


EpisodeIndices = SampleIndices


@dataclass
class VideoGeometry:
    frames: np.ndarray
    depths: np.ndarray
    intrinsics: np.ndarray
    poses_c2w: np.ndarray
    masks: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None
    original_size: Optional[tuple[int, int]] = None  # height and width before geometry processing
    processed_size: Optional[tuple[int, int]] = None  # height and width used for geometry processing
