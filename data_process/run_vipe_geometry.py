#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np


ARTIFACT_PATHS = {
    "pose.npz": Path("pose/clip.npz"),
    "depth.zip": Path("depth/clip.zip"),
    "intrinsics.npz": Path("intrinsics/clip.npz"),
    "mask.zip": Path("mask/clip.zip"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViPE on prepared clips and import its outputs."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--vipe-executable", type=Path, default=Path("vipe"))
    parser.add_argument("--pipeline", default="dav3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temp-root", type=Path, default=None)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def list_sample_dirs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and len(path.name) == 8 and path.name.isdigit()
    )


def video_shape(video_path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")
    return frame_count, height, width


def validate_npz(path: Path, frame_count: int, expected_shape: tuple[int, ...]) -> None:
    with np.load(path) as payload:
        inds = np.asarray(payload["inds"])
        data = np.asarray(payload["data"])
    if not np.array_equal(inds, np.arange(frame_count)):
        raise ValueError(f"Frame indices are incomplete in {path}: {inds.shape}")
    if data.shape != expected_shape:
        raise ValueError(
            f"Unexpected data shape in {path}: expected {expected_shape}, got {data.shape}"
        )


def validate_zip(path: Path, frame_count: int, suffix: str) -> None:
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist() if name.lower().endswith(suffix)
        )
    expected = [f"{index:05d}{suffix}" for index in range(frame_count)]
    if [Path(name).name for name in members] != expected:
        raise ValueError(
            f"Unexpected frame members in {path}: expected {frame_count}, got {len(members)}"
        )


def validate_artifacts(result_root: Path, clip_path: Path) -> None:
    frame_count, _height, _width = video_shape(clip_path)
    paths = {name: result_root / relative for name, relative in ARTIFACT_PATHS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ViPE did not produce required artifacts: {missing}")
    validate_npz(paths["pose.npz"], frame_count, (frame_count, 4, 4))
    validate_npz(paths["intrinsics.npz"], frame_count, (frame_count, 4))
    validate_zip(paths["depth.zip"], frame_count, ".exr")
    validate_zip(paths["mask.zip"], frame_count, ".png")


def copy_artifacts(result_root: Path, sample_dir: Path) -> None:
    for output_name, relative_path in ARTIFACT_PATHS.items():
        source = result_root / relative_path
        destination = sample_dir / output_name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def has_complete_artifacts(sample_dir: Path) -> bool:
    return all((sample_dir / name).is_file() for name in ARTIFACT_PATHS)


def main() -> None:
    args = parse_args()
    sample_dirs = list_sample_dirs(args.input_root)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        sample_dirs = sample_dirs[: args.limit]

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed rank: rank={rank}, world_size={world_size}")
    sample_dirs = sample_dirs[rank::world_size]

    child_env = os.environ.copy()
    visible_devices = child_env.get("CUDA_VISIBLE_DEVICES")
    if world_size > 1:
        if visible_devices:
            devices = [device.strip() for device in visible_devices.split(",")]
            child_env["CUDA_VISIBLE_DEVICES"] = devices[local_rank % len(devices)]
        else:
            child_env["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)

    print(
        f"ViPE shard: rank={rank}/{world_size}, local_rank={local_rank}, "
        f"samples={len(sample_dirs)}",
        flush=True,
    )

    processed = 0
    skipped = 0
    temp_parent = str(args.temp_root) if args.temp_root is not None else None
    failed: list[str] = []
    failed_log = args.input_root / f"_vipe_failed_rank{rank}.txt"
    for sample_dir in sample_dirs:
        clip_path = sample_dir / "clip.mp4"
        if not clip_path.is_file():
            raise FileNotFoundError(clip_path)
        if args.skip_existing and has_complete_artifacts(sample_dir):
            skipped += 1
            continue

        # retry each clip once, then record the failure and continue
        done = False
        for attempt in (1, 2):
            with tempfile.TemporaryDirectory(
                prefix=f"vipe-{sample_dir.name}-",
                dir=temp_parent,
            ) as temporary_dir:
                result_root = Path(temporary_dir) / "results"
                command = [
                    str(args.vipe_executable),
                    "infer",
                    str(clip_path),
                    "--output",
                    str(result_root),
                    "--pipeline",
                    args.pipeline,
                ]
                print("Running:", " ".join(command), flush=True)
                try:
                    subprocess.run(command, check=True, env=child_env)
                    validate_artifacts(result_root, clip_path)
                    copy_artifacts(result_root, sample_dir)
                    done = True
                except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as exc:
                    print(
                        f"ViPE attempt {attempt} FAILED for {sample_dir.name}: {exc}",
                        flush=True,
                    )
            if done:
                break
        if done:
            processed += 1
            print(f"Imported ViPE geometry for {sample_dir}", flush=True)
        else:
            failed.append(sample_dir.name)
            with failed_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{sample_dir.name}\n")

    print(
        f"ViPE geometry complete: processed={processed}, skipped={skipped}, "
        f"failed={len(failed)}{' -> ' + str(failed_log) if failed else ''}"
    )


if __name__ == "__main__":
    main()
