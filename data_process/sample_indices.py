from typing import Optional

import numpy as np

from data_process.types import SampleIndices


REF_CANDIDATE_SCOPES = ("two_sided", "past_only")


def filter_reference_candidates(
    candidate_indices: list[int],
    preceding_indices: list[int],
    scope: str,
) -> list[int]:
    if scope not in REF_CANDIDATE_SCOPES:
        raise ValueError(
            f"Unknown ref_candidate_scope {scope!r}; expected one of "
            f"{REF_CANDIDATE_SCOPES}."
        )
    if scope == "two_sided":
        return list(candidate_indices)
    cutoff = min(preceding_indices)
    return [i for i in candidate_indices if i < cutoff]


def sample_frame_indices(
    num_frames: int,
    N_target: int,
    M_pre: int,
    min_gap_for_candidates: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> SampleIndices:
    # take preceding frames before t0 and target frames from t0 onward
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if N_target <= 0 or M_pre <= 0:
        raise ValueError("N_target and M_pre must be positive")
    if num_frames < N_target + M_pre:
        raise ValueError("Not enough frames to sample training sample")

    if rng is None:
        rng = np.random.default_rng()

    t0_min = M_pre
    t0_max = num_frames - N_target
    if t0_min > t0_max:
        raise ValueError("Invalid sampling range for t0")

    t0 = int(rng.integers(t0_min, t0_max + 1))
    target_indices = list(range(t0, t0 + N_target))
    preceding_indices = list(range(t0 - M_pre, t0))

    preceding_set = set(preceding_indices)
    target_set = set(target_indices)
    candidate_indices = [
        i for i in range(num_frames) if i not in preceding_set and i not in target_set
    ]

    if min_gap_for_candidates > 0:
        protected = sorted(preceding_indices + target_indices)

        def far_enough(idx: int) -> bool:
            return min(abs(idx - t) for t in protected) >= min_gap_for_candidates

        candidate_indices = [i for i in candidate_indices if far_enough(i)]

    return SampleIndices(
        t0=t0,
        preceding_indices=preceding_indices,
        target_indices=target_indices,
        candidate_indices=candidate_indices,
    )


sample_episode_indices = sample_frame_indices
