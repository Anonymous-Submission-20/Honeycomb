import os
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def get_rank_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    gpu_count = int(os.environ.get("HONEYCOMB_GPU_COUNT", "0"))
    if gpu_count > 0:
        local_rank %= gpu_count
    return rank, world_size, local_rank


def shard_items(
    items: Sequence[T], rank: int, world_size: int, mode: str = "interleave"
) -> list[T]:
    if world_size <= 1:
        return list(items)

    if mode == "contiguous":
        n = len(items)
        base_size = n // world_size
        remainder = n % world_size
        # give the first few ranks one extra item when the split is uneven
        if rank < remainder:
            start = rank * (base_size + 1)
            end = start + base_size + 1
        else:
            start = remainder * (base_size + 1) + (rank - remainder) * base_size
            end = start + base_size
        return list(items[start:end])
    else:
        return [item for idx, item in enumerate(items) if idx % world_size == rank]
