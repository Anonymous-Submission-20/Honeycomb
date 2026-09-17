import os

os.environ.setdefault("LOG_LEVEL", "WARNING")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


@dataclass
class DataConfig:
    video_dirs: list[str] = field(
        default_factory=lambda: [
            path
            for path in os.environ.get(
                "HONEYCOMB_VIDEO_DIRS",
                "data/videos",
            ).split(os.pathsep)
            if path
        ]
    )
    output_root: str = os.environ.get("HONEYCOMB_OUTPUT_ROOT", "data/train")

    clip_num_frames: int = 81
    clip_target_fps: int = 16
    clip_target_width: int = 1280
    clip_target_height: int = 704

    max_videos: Optional[int] = _env_optional_int("HONEYCOMB_MAX_VIDEOS", 100)
    shuffle_seed: Optional[int] = _env_optional_int("HONEYCOMB_SHUFFLE_SEED", 41)
    fps_override: Optional[float] = None
    naming_style: str = "figure"
    skip_existing: bool = True
    quiet_nonzero: bool = True
    cleanup_interval: int = 50

    video_vae_model_path: str = "data/Wan-AI/Wan2.2-TI2V-5B"
    video_vae_checkpoint: str = "Wan2.2_VAE.pth"

    N_target: int = 33
    M_pre: int = 8
    min_gap_for_candidates: int = 2
    K_ref_stride: int = 2
    # use an IoU threshold of 0.04 for RealEstate10K and 0.01 for SpatialVID
    eps_iou: float = float(os.environ.get("HONEYCOMB_EPS_IOU", 0.04))
    max_refs: int = 8
    # only use references before the preceding-frame window
    ref_candidate_scope: str = os.environ.get(
        "HONEYCOMB_REF_CANDIDATE_SCOPE", "past_only"
    )
    # set HONEYCOMB_APPLY_DYNAMIC_MASK=0 to keep dynamic objects and sky
    apply_dynamic_mask: bool = os.environ.get(
        "HONEYCOMB_APPLY_DYNAMIC_MASK", "1") not in ("0", "false", "False")
    point_cloud_vae_model_path: str = "data/Wan-AI/Wan2.2-TI2V-5B"
    point_cloud_vae_checkpoint: str = "Wan2.2_VAE.pth"
    point_cloud_vae_dtype: str = "bf16"
    scene_voxel_size: float = 0.01  # reference selection uses voxels ten times larger
    num_samples: int = 1
    sample_random_seed: Optional[int] = None

    def print_config(self) -> None:
        print("=" * 60)
        print("LSM Pipeline Configuration")
        print("=" * 60)
        print(f"Video dirs      : {self.video_dirs}")
        print(f"Output root     : {self.output_root}")
        print(f"Max videos      : {self.max_videos}")
        print(f"Skip existing   : {self.skip_existing}")
        print("--- Clip Extraction ---")
        print(f"  Num frames    : {self.clip_num_frames}")
        print(f"  Target FPS    : {self.clip_target_fps}")
        print(f"  Resolution    : {self.clip_target_width}x{self.clip_target_height}")
        print("--- Video VAE ---")
        print(f"  Model path    : {self.video_vae_model_path}")
        print(f"  Checkpoint    : {self.video_vae_checkpoint}")
        print("--- Sample ---")
        print(f"  N_target      : {self.N_target}")
        print(f"  M_pre         : {self.M_pre}")
        print(f"  eps_iou       : {self.eps_iou}")
        print(f"  max_refs      : {self.max_refs}")
        print(f"  ref_scope     : {self.ref_candidate_scope}")
        print(f"  point_vae_dty : {self.point_cloud_vae_dtype}")
        print(f"  voxel_size    : {self.scene_voxel_size}")
        print("=" * 60)


CONFIG = DataConfig()


def load_config(path: str | Path | None = None) -> DataConfig:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return CONFIG

    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML required for YAML configs") from exc
        data = yaml.safe_load(path.read_text())
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary")

    return DataConfig(**{k: v for k, v in data.items() if hasattr(DataConfig, k)})


@dataclass
class SampleConfig:
    N_target: int = CONFIG.N_target
    M_pre: int = CONFIG.M_pre
    min_gap_for_candidates: int = CONFIG.min_gap_for_candidates
    K_ref_stride: int = CONFIG.K_ref_stride
    eps_iou: float = CONFIG.eps_iou
    max_refs: int = CONFIG.max_refs
    ref_candidate_scope: str = CONFIG.ref_candidate_scope
    apply_dynamic_mask: bool = CONFIG.apply_dynamic_mask
    point_cloud_vae_model_path: str = CONFIG.point_cloud_vae_model_path
    point_cloud_vae_checkpoint: str = CONFIG.point_cloud_vae_checkpoint
    point_cloud_vae_dtype: str = CONFIG.point_cloud_vae_dtype
    scene_voxel_size: float = CONFIG.scene_voxel_size
    ref_iou_voxel_size: Optional[float] = None  # use scene_voxel_size * 10 when unset
    num_samples: int = CONFIG.num_samples
    random_seed: Optional[int] = CONFIG.sample_random_seed


EpisodeConfig = SampleConfig


def get_sample_config() -> SampleConfig:
    return SampleConfig()


get_episode_config = get_sample_config
