# latent frame j corresponds to clip frame 4 * j
LATENT_STRIDE = 4

# memory sizes read at the start of each chunk in a padded 81-frame rollout
ROLLOUT_RUNGS = (1, 9, 17)


def latent_frame_indices(num_frames: int, stride: int = LATENT_STRIDE) -> list[int]:
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    return list(range(0, num_frames, stride))


def num_real_times(num_frames: int, stride: int = LATENT_STRIDE) -> int:
    return len(latent_frame_indices(num_frames, stride))


def normalized_time(t: float, num_time_steps: int) -> float:
    if num_time_steps < 2:
        raise ValueError(f"num_time_steps must be >= 2, got {num_time_steps}")
    return 2.0 * float(t) / float(num_time_steps - 1) - 1.0


def rollout_time_bounds(num_time_steps: int,
                        num_real: int | None = None) -> tuple[float, float]:
    if num_time_steps < 2:
        raise ValueError(f"num_time_steps must be >= 2, got {num_time_steps}")
    if num_real is not None and num_real > num_time_steps:
        raise ValueError(
            f"num_real_times ({num_real}) exceeds num_time_steps "
            f"({num_time_steps}); the field has no slot for the extra times")
    return (0.0, float(num_time_steps - 1))


def rollout_write_sets(num_real: int,
                       rungs: tuple[int, ...] = ROLLOUT_RUNGS) -> list[list[int]]:
    out = []
    for n in rungs:
        if n < 1:
            raise ValueError(f"rung {n} must be >= 1")
        if n > num_real:
            raise ValueError(
                f"rung {n} exceeds the {num_real} real latent times available")
        out.append(list(range(n)))
    return out
