MAX_FRAMES_PER_ITERATION = 33


def validate_num_frames(num_frames: int) -> bool:
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if (num_frames - 1) % 4 != 0:
        valid_examples = [4 * n + 1 for n in range(1, 12)]
        raise ValueError(
            f"num_frames ({num_frames}) must be 4N+1 format.\nValid examples: {valid_examples}, ..."
        )
    return True


def compute_iteration_plan(num_frames: int) -> list:
    validate_num_frames(num_frames)
    plan = []
    if num_frames <= MAX_FRAMES_PER_ITERATION:
        plan.append((0, num_frames, num_frames))
    else:
        plan.append((0, MAX_FRAMES_PER_ITERATION, MAX_FRAMES_PER_ITERATION))
        current_output_frame = MAX_FRAMES_PER_ITERATION
        remaining = num_frames - MAX_FRAMES_PER_ITERATION
        while remaining > 0:
            new_frames_this_iter = min(32, remaining)
            model_frames = new_frames_this_iter + 1
            if (model_frames - 1) % 4 != 0:
                model_frames = ((model_frames - 1) // 4 + 1) * 4 + 1
            output_start = current_output_frame
            output_end = current_output_frame + new_frames_this_iter
            plan.append((output_start, output_end, model_frames))
            current_output_frame = output_end
            remaining -= new_frames_this_iter
    return plan
