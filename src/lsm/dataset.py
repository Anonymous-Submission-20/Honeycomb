import io
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import lmdb
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    data_path: Path = Path("data")

    random_sample_ref: bool = False
    random_sample_preceding: bool = False
    max_reference_frames: int | None = None
    max_preceding_frames: int | None = None


class LSMDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        model_version: str,
    ):
        self.config: DataConfig = config
        self.data_path = config.data_path
        self.model_version = model_version
        assert self.model_version in {"14b", "5b"}, (
            f"model_version must be either '14b' or '5b', got {model_version}"
        )

        self.random_sample_ref = config.random_sample_ref
        self.random_sample_preceding = config.random_sample_preceding
        self.max_reference_frames = config.max_reference_frames
        self.max_preceding_frames = config.max_preceding_frames

        self.shard_paths = []
        self.index = []

        for path in sorted(self.data_path.iterdir()):
            if path.is_dir() and path.name.endswith(".lmdb"):
                self.shard_paths.append(str(path))

        assert self.shard_paths, f"No LMDB shards found in {self.data_path}"

        # index shard lengths without loading every sample key
        self.envs = None
        self.shard_lengths = []
        self.shard_offsets = [0]

        for path in self.shard_paths:
            tmp_env = self._open_lmdb_env(path)
            with tmp_env.begin() as txn:
                num_entries = txn.stat()["entries"]
            tmp_env.close()

            self.shard_lengths.append(num_entries)
            self.shard_offsets.append(self.shard_offsets[-1] + num_entries)

        self.total_samples = self.shard_offsets[-1]

        print(
            f"Loaded {self.total_samples} samples from {len(self.shard_paths)} shards"
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._getitem(idx)
        except KeyError:
            print(f"WARNING MISSING DATA FOR KEY {idx}")
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            return self._getitem(idx - 1)

    def _getitem(self, idx: int) -> dict:
        self.reopen_envs()

        shard_id = 0
        for i, offset in enumerate(self.shard_offsets[1:], start=0):
            if idx < offset:
                shard_id = i
                break

        local_idx = idx - self.shard_offsets[shard_id]

        with self.envs[shard_id].begin() as txn:
            cursor = txn.cursor()
            if not cursor.first():
                raise KeyError(f"Empty shard {shard_id}")

            for _ in range(local_idx):
                if not cursor.next():
                    raise KeyError(
                        f"Index {local_idx} out of range in shard {shard_id}"
                    )

            key, raw_data = cursor.item()
            if raw_data is None:
                raise KeyError(f"Key {key} not found in shard {shard_id}")

            sample = pickle.loads(raw_data)

        target_latent = self._load_tensor_from_bytes(sample["train_target_rgb.pt"])
        latent_channels = target_latent.shape[1]
        target_scene_proj = self._load_tensor_from_bytes(
            sample["train_target_scene_proj_rgb.pt"]
        )
        preceding_latent_raw = self._load_tensor_from_bytes(
            sample["train_preceding_rgb.pt"]
        )
        preceding_scene_proj_raw = self._load_tensor_from_bytes(
            sample["train_preceding_scene_proj_rgb.pt"]
        )

        if "train_reference_rgb.pt" in sample:
            reference_latent_raw = self._load_tensor_from_bytes(
                sample["train_reference_rgb.pt"]
            )
        else:
            H, W = target_latent.shape[-2:]
            reference_latent_raw = self._create_empty_reference_latent_raw(
                latent_channels=latent_channels,
                height=H,
                width=W,
                dtype=target_latent.dtype,
            )

        # apply the same preceding-frame selection to latents and scene projections
        preceding_latent, preceding_scene_proj = self._process_preceding_latent(
            preceding_latent_raw, preceding_scene_proj_raw
        )

        reference_latent = self._process_reference_latent(
            reference_latent_raw, latent_channels
        )

        prompt = (
            sample["train_target_rgb.txt"].decode()
            if isinstance(sample["train_target_rgb.txt"], bytes)
            else sample["train_target_rgb.txt"]
        )

        img_bytes = sample["train_target_rgb_frame0.png"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        meta = (
            json.loads(sample["train_sample.json"])
            if isinstance(sample["train_sample.json"], bytes)
            else sample["train_sample.json"]
        )

        batch = {
            "idx": idx,
            "prompts": prompt,
            "target_latent": target_latent.to(dtype=torch.float32),
            "target_scene_proj": target_scene_proj.to(
                dtype=torch.float32
            ),
            "preceding_scene_proj": preceding_scene_proj.to(
                dtype=torch.float32
            ),
            "preceding_latent": preceding_latent.to(
                dtype=torch.float32
            ),
            "reference_latent": reference_latent.to(
                dtype=torch.float32
            ),
            "img": img,
            "meta": meta,
        }

        return batch

    def reopen_envs(self):
        if self.envs is not None:
            return
        self.envs = []
        for path in self.shard_paths:
            self.envs.append(self._open_lmdb_env(path))

    def _open_lmdb_env(self, path: str) -> lmdb.Environment:
        return lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,  # disable read-ahead because samples are accessed randomly
            meminit=False,
            max_readers=2048,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["envs"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.envs = None

    def _load_tensor_from_bytes(self, data: bytes) -> torch.Tensor:
        tensor_data = torch.load(io.BytesIO(data), weights_only=False)
        if isinstance(tensor_data, dict):
            return tensor_data["latent"]
        return tensor_data

    def _create_empty_reference_latent_raw(
        self, latent_channels: int, height: int, width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.model_version == "14b":
            return torch.zeros(1, 0, height, width, dtype=dtype)
        return torch.zeros(latent_channels, 0, height, width, dtype=dtype)

    def _empty_frame_tensor_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(0, *tensor.shape[1:], dtype=tensor.dtype)

    def _cap_preceding_latent(
        self, preceding_latent: torch.Tensor, preceding_scene_proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_preceding_frames is None:
            return preceding_latent, preceding_scene_proj
        if self.max_preceding_frames == 0:
            return (
                self._empty_frame_tensor_like(preceding_latent),
                self._empty_frame_tensor_like(preceding_scene_proj),
            )
        return (
            preceding_latent[-self.max_preceding_frames :],
            preceding_scene_proj[-self.max_preceding_frames :],
        )

    def _process_preceding_latent(
        self, preceding_latent: torch.Tensor, preceding_scene_proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preceding_latent, preceding_scene_proj = self._cap_preceding_latent(
            preceding_latent, preceding_scene_proj
        )

        if not self.random_sample_preceding:
            return preceding_latent, preceding_scene_proj

        num_frames = preceding_latent.shape[0]
        if num_frames <= 0:
            return preceding_latent, preceding_scene_proj

        # allow zero frames so sampling can drop all preceding context
        num_sampled = random.randint(0, num_frames)

        if num_sampled == 0:
            C, H, W = preceding_latent.shape[1:]
            return (
                torch.zeros(0, C, H, W, dtype=preceding_latent.dtype),
                torch.zeros(0, C, H, W, dtype=preceding_scene_proj.dtype),
            )

        if num_sampled < num_frames:
            # keep latent and scene-projection indices aligned
            indices = sorted(random.sample(range(num_frames), num_sampled))
            preceding_latent = preceding_latent[indices]
            preceding_scene_proj = preceding_scene_proj[indices]

        return preceding_latent, preceding_scene_proj

    def _process_reference_latent(
        self, reference_latent_raw: torch.Tensor, latent_channels: int
    ) -> torch.Tensor:
        if self.model_version == "14b":
            reference_frames = self._process_reference_latent_14b(
                reference_latent_raw, latent_channels
            )
        else:
            reference_frames = self._process_reference_latent_5b(
                reference_latent_raw, latent_channels
            )

        return self._process_reference_frames(reference_frames)

    def _process_reference_latent_14b(
        self, reference_latent_raw: torch.Tensor, latent_channels: int
    ) -> torch.Tensor:
        _, total_channels, H, W = reference_latent_raw.shape
        if total_channels % latent_channels != 0:
            raise ValueError(
                "Reference latent channels do not match target latent channels: "
                f"total_channels={total_channels}, latent_channels={latent_channels}"
            )

        num_ref_frames = total_channels // latent_channels

        if num_ref_frames <= 0:
            return torch.zeros(
                0, latent_channels, H, W, dtype=reference_latent_raw.dtype
            )

        reference_frames = reference_latent_raw.squeeze(0).view(
            num_ref_frames, latent_channels, H, W
        )

        return reference_frames

    def _process_reference_latent_5b(
        self, reference_latent_raw: torch.Tensor, latent_channels: int
    ) -> torch.Tensor:
        if reference_latent_raw.ndim != 4:
            raise ValueError(
                "5B reference latent must have shape [C, R, H, W], "
                f"got shape {tuple(reference_latent_raw.shape)}"
            )

        channels, num_ref_frames, H, W = reference_latent_raw.shape
        if channels != latent_channels:
            raise ValueError(
                "5B reference latent channels do not match target latent channels: "
                f"channels={channels}, latent_channels={latent_channels}"
            )

        if num_ref_frames <= 0:
            return torch.zeros(
                0, latent_channels, H, W, dtype=reference_latent_raw.dtype
            )

        reference_frames = reference_latent_raw.permute(1, 0, 2, 3)

        return reference_frames

    def _process_reference_frames(self, reference_frames: torch.Tensor) -> torch.Tensor:
        if self.max_reference_frames is not None:
            reference_frames = reference_frames[: self.max_reference_frames]

        if not self.random_sample_ref:
            return reference_frames

        num_ref_frames = reference_frames.shape[0]
        if num_ref_frames <= 0:
            return reference_frames

        num_sampled = random.randint(0, num_ref_frames)
        if num_sampled == 0:
            return self._empty_frame_tensor_like(reference_frames)

        if num_sampled < num_ref_frames:
            indices = sorted(random.sample(range(num_ref_frames), num_sampled))
            reference_frames = reference_frames[indices]

        return reference_frames


def _pad_variable_length_tensors(
    tensors: list, name: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    counts = [t.shape[0] for t in tensors]

    if len(set(counts)) == 1:
        return torch.stack(tensors), None

    max_count = max(counts)

    if max_count == 0:
        return torch.stack(tensors), None

    for t in tensors:
        if t.shape[0] > 0:
            C, H, W = t.shape[1:]
            dtype = t.dtype
            break
    else:
        return torch.stack(tensors), None

    padded = []
    masks = []

    for t in tensors:
        num = t.shape[0]
        if num < max_count:
            if num == 0:
                padded_t = torch.zeros(max_count, C, H, W, dtype=dtype)
            else:
                padding = torch.zeros(max_count - num, C, H, W, dtype=dtype)
                padded_t = torch.cat([t, padding], dim=0)
        else:
            padded_t = t
        padded.append(padded_t)

        mask = torch.zeros(max_count, dtype=torch.bool)
        mask[:num] = True
        masks.append(mask)

    return torch.stack(padded), torch.stack(masks)


def lsm_collate_fn(batch):
    collated = {}

    collated["idx"] = torch.tensor([b["idx"] for b in batch])
    collated["prompts"] = [b["prompts"] for b in batch]
    collated["meta"] = [b["meta"] for b in batch]

    collated["target_latent"] = torch.stack([b["target_latent"] for b in batch])
    collated["target_scene_proj"] = torch.stack([b["target_scene_proj"] for b in batch])
    collated["img"] = torch.stack([b["img"] for b in batch])

    preceding_latent, preceding_mask = _pad_variable_length_tensors(
        [b["preceding_latent"] for b in batch], "preceding_latent"
    )
    preceding_scene_proj, _ = _pad_variable_length_tensors(
        [b["preceding_scene_proj"] for b in batch], "preceding_scene_proj"
    )
    collated["preceding_latent"] = preceding_latent
    collated["preceding_scene_proj"] = preceding_scene_proj
    if preceding_mask is not None:
        collated["preceding_mask"] = preceding_mask

    reference_latent, reference_mask = _pad_variable_length_tensors(
        [b["reference_latent"] for b in batch], "reference_latent"
    )
    collated["reference_latent"] = reference_latent
    if reference_mask is not None:
        collated["reference_mask"] = reference_mask

    return collated
