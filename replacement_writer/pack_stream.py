from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


class PackStream:
    def __init__(self, paths, loader, rung_fn=None, n_rungs=None,
                 num_workers=4, prefetch=None, shuffle=True, seed=0):
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise ValueError("PackStream got no packs")
        if rung_fn is None and n_rungs is None:
            raise ValueError("give either rung_fn or n_rungs")
        self.loader = loader
        self.rung_fn = rung_fn
        self.n_rungs = n_rungs
        self.num_workers = max(1, int(num_workers))
        self.prefetch = int(prefetch) if prefetch else 2 * self.num_workers
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def _load_one(self, path, rung):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        return self.loader(raw, rung=rung), path, rung

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        # use one generator per epoch so clip order and rung choices stay reproducible
        g = torch.Generator().manual_seed(self.seed + 9973 * self.epoch)
        n = len(self.paths)
        order = (torch.randperm(n, generator=g).tolist() if self.shuffle
                 else list(range(n)))
        rungs = [self.rung_fn(i) if self.rung_fn is not None
                 else int(torch.randint(0, self.n_rungs, (1,), generator=g).item())
                 for i in order]
        self.epoch += 1

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            pending, cursor = deque(), 0
            while cursor < n and len(pending) < self.prefetch:
                pending.append(ex.submit(self._load_one,
                                         self.paths[order[cursor]], rungs[cursor]))
                cursor += 1
            while pending:
                fut = pending.popleft()
                if cursor < n:
                    pending.append(ex.submit(self._load_one,
                                             self.paths[order[cursor]], rungs[cursor]))
                    cursor += 1
                yield fut.result()


def pack_paths(dirs):
    out = []
    for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs):
        found = sorted(Path(d).glob("*.pt"))
        if not found:
            raise ValueError(f"no packs under {d}")
        out.extend(str(p) for p in found)
    return sorted(set(out))
