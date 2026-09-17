from __future__ import annotations
import json
import sys
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import Tensor
from tqdm.auto import tqdm
from lsm.inference.utils import MAX_FRAMES_PER_ITERATION as MAX_CHUNK
from lsm.inference.utils import compute_iteration_plan
from lsm.latent_point_cloud import LatentPointCloud
from lsm.spatia.flow_match import FlowUniPCMultistepScheduler
from lsm.spatia.utils import load_state_dict
from lsm.spatia.visibility import append_visibility_channel
from lsm.spatia.wan_video_new import WanVideoPipeline, model_fn_wan_video


@dataclass
class VideoGeometry:
    frames: np.ndarray
    depths: np.ndarray
    intrinsics: np.ndarray
    poses_c2w: np.ndarray
    masks: np.ndarray | None = None
    frame_indices: np.ndarray | None = None
    original_size: tuple[int, int] | None = None
    processed_size: tuple[int, int] | None = None


@dataclass
class InferenceConfig:
    num_frames: int = 33
    start_frame: int = 0
    infer_steps: int = 40
    num_train_timesteps: int = 1000
    timestep_shift: float = 5.0
    guidance_scale: float = 1.0
    no_cfg: bool = True
    negative_prompt: str = ""
    prompt_suffix: str = ""
    fps: int = 16
    max_reference_frames: int = 8
    preceding_pixel_frames: int = 8
    seed: int = 42
    height: int | None = None
    width: int | None = None
    tiled: bool = False
    lora_on_first_chunk: bool = False
    exclude_recent_refs: bool = True
    tile_size: tuple[int, int] = (30, 52)
    tile_stride: tuple[int, int] = (15, 26)
    ref_iou_threshold: float = 0.04
    ref_iou_voxel_size: float = 0.1
    update_memory: bool = True
    preceding_first: bool = True
    pad_final_chunk: bool = True
    reencode_preceding: bool = True
    reencode_anchor: bool = True
    reencode_references: bool = True
    depth_python: str = sys.executable


def load_vace_checkpoint(pipe: WanVideoPipeline, path: Path) -> None:
    checkpoint = open_checkpoint(path)
    state = extract_vace_state_dict(checkpoint)
    if not state:
        raise ValueError(f"No VACE tensors found in {path}.")
    target_state = pipe.vace.state_dict()
    ckpt_pe = state.get("vace_patch_embedding.weight")
    model_pe = target_state.get("vace_patch_embedding.weight")
    if (
        ckpt_pe is not None
        and model_pe is not None
        and (ckpt_pe.shape[1] != model_pe.shape[1])
    ):
        raise ValueError(
            f"VACE input-channel mismatch: checkpoint {path} has vace_in_dim={int(ckpt_pe.shape[1])} but the model was built with vace_in_dim={int(model_pe.shape[1])}. Honeycomb requires a 49-channel VACE checkpoint."
        )
    compatible = {}
    skipped = 0
    for key, value in state.items():
        if key not in target_state or value.shape != target_state[key].shape:
            skipped += 1
            continue
        compatible[key] = value.to(dtype=target_state[key].dtype)
    missing, unexpected = pipe.vace.load_state_dict(compatible, strict=False)
    print(
        f"Loaded VACE checkpoint: {path} tensors={len(compatible)} skipped={skipped} missing={len(missing)} unexpected={len(unexpected)}"
    )


def open_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        return dict(load_safetensors(str(path), device="cpu"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    return checkpoint


def extract_vace_state_dict(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
    candidate = checkpoint
    if isinstance(checkpoint.get("vace"), dict):
        candidate = checkpoint["vace"]
    elif isinstance(checkpoint.get("generator"), dict) and isinstance(
        checkpoint["generator"].get("vace"), dict
    ):
        candidate = checkpoint["generator"]["vace"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        candidate = checkpoint["state_dict"]
    state = {}
    for key, value in candidate.items():
        if not torch.is_tensor(value):
            continue
        normalized = strip_prefixes(key, ("module.", "model.", "pipe.vace.", "vace."))
        if normalized.startswith(("vace_blocks.", "vace_patch_embedding.")):
            state[normalized] = value
    return state


def strip_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def validate_pipe(pipe: WanVideoPipeline) -> None:
    if pipe.dit is None:
        raise ValueError("WanVideoPipeline did not load a DiT model.")
    if pipe.vace is None:
        raise ValueError("WanVideoPipeline did not load a VACE model.")
    if not getattr(pipe.dit, "fuse_vae_embedding_in_latents", False):
        raise ValueError()
    if not getattr(pipe.dit, "seperated_timestep", False):
        raise ValueError()
    patch_embedding = getattr(pipe.vace, "vace_patch_embedding", None)
    if patch_embedding is None:
        raise ValueError()
    if int(patch_embedding.in_channels) != 49:
        raise ValueError(
            f"Unsupported VACE input width {int(patch_embedding.in_channels)}; expected 49."
        )


class HoneycombPipeline:
    def __init__(
        self, pipe: WanVideoPipeline, config: InferenceConfig, memory_factory=None
    ) -> None:
        self.pipe = pipe
        self.config = config
        self.lora_state_dict: dict | None = None
        self.lora_alpha: float = 1.0
        self.memory_factory = memory_factory

    def build_seed_memory(
        self,
        *,
        geometry: VideoGeometry,
        first_frame_latent: Tensor,
        gen_frames: int,
        device: torch.device,
        temporal_stride: int = 4,
    ):
        start = self.config.start_frame
        initial_mask = get_initial_exclusion_mask(geometry, start)
        if self.memory_factory is None:
            return LatentPointCloud.from_geometry(
                depth=geometry.depths[start],
                intrinsics=geometry.intrinsics[start],
                cam2world=geometry.poses_c2w[start],
                latent=first_frame_latent[0],
                mask=initial_mask,
                device=device,
            )
        return self.memory_factory(
            depth=geometry.depths[start],
            intrinsics=geometry.intrinsics[start],
            cam2world=geometry.poses_c2w[start],
            latent=first_frame_latent[0],
            mask=initial_mask,
            device=device,
            num_time_steps=(gen_frames - 1) // temporal_stride + 1,
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        geometry_path: Path,
        prompt: str,
        output_dir: Path,
        run_metadata: dict[str, Any] | None = None,
    ) -> Tensor:
        geometry = load_video_geometry_for_inference(
            geometry_path, start_frame=self.config.start_frame
        )
        requested_frames = self.config.num_frames
        gen_frames = requested_frames
        if self.config.pad_final_chunk:
            gen_frames = full_chunk_length(requested_frames)
            need = self.config.start_frame + gen_frames
            if gen_frames != requested_frames and len(geometry.poses_c2w) < need:
                geometry = pad_geometry_to(geometry, need)
                print(
                    f"pad_final_chunk: {requested_frames} -> {gen_frames} frames, geometry padded to {len(geometry.poses_c2w)} poses"
                )
        validate_geometry(geometry, start_frame=self.config.start_frame)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device(self.pipe.device)
        dtype = self.pipe.torch_dtype
        height, width = resolve_output_hw(geometry, self.config)
        temporal_stride = 4
        preceding_latent_frames = (
            self.config.preceding_pixel_frames + temporal_stride - 1
        ) // temporal_stride
        iteration_plan = compute_iteration_plan(gen_frames)
        self.pipe.load_models_to_device(["vae"])
        first_frame = resize_frame(
            geometry.frames[self.config.start_frame], height, width
        )
        first_frame_latent = encode_video_frames(
            self.pipe,
            first_frame[None],
            tiled=self.config.tiled,
            tile_size=self.config.tile_size,
            tile_stride=self.config.tile_stride,
        ).to(device=device, dtype=dtype)
        lpc = self.build_seed_memory(
            geometry=geometry,
            first_frame_latent=first_frame_latent,
            gen_frames=gen_frames,
            device=device,
            temporal_stride=temporal_stride,
        )
        pos_prompt = prompt
        if self.config.prompt_suffix:
            pos_prompt = f"{prompt} {self.config.prompt_suffix}".strip()
        context = self.encode_prompt(pos_prompt, positive=True)
        uncon_context = None
        if not self.config.no_cfg:
            uncon_context = self.encode_prompt(
                self.config.negative_prompt, positive=False
            )
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed)
        generated_latents: list[Tensor] = []
        generated_scene_latents: list[Tensor] = []
        generated_frames: list[np.ndarray] = []
        frame_visible_points: dict[int, np.ndarray] = {}
        latent_to_frame: dict[int, int] = {}
        metadata: dict[str, Any] = {
            "config": asdict(self.config),
            "geometry_path": str(geometry_path),
            "height": height,
            "width": width,
            "iterations": [],
            "prompt_positive": pos_prompt,
            "prompt_negative": self.config.negative_prompt
            if not self.config.no_cfg
            else None,
        }
        if run_metadata is not None:
            metadata.update(run_metadata)
        metadata["rollout_depth_model"] = "da3"
        metadata["write_decode"] = "joint"
        for iter_idx, (output_start, output_end, model_frames) in enumerate(
            iteration_plan
        ):
            if self.lora_state_dict is not None:
                if iter_idx == 0 and (not self.config.lora_on_first_chunk):
                    self.pipe.unload_lora(
                        self.pipe.dit,
                        lora_state_dict=self.lora_state_dict,
                        alpha=self.lora_alpha,
                    )
                else:
                    self.pipe.load_lora(
                        self.pipe.dit,
                        lora_state_dict=self.lora_state_dict,
                        alpha=self.lora_alpha,
                    )
            target_pose_indices = build_target_pose_indices(
                start_frame=self.config.start_frame,
                output_start=output_start,
                model_frames=model_frames,
                temporal_stride=temporal_stride,
                iter_idx=iter_idx,
            )
            if target_pose_indices[-1] >= len(geometry.poses_c2w):
                raise ValueError(
                    f"Target pose index exceeds geometry length: {target_pose_indices[-1]} >= {len(geometry.poses_c2w)}"
                )
            target_scene = project_lpc_sequence(
                lpc=lpc, geometry=geometry, frame_indices=target_pose_indices
            )
            preceding_latents, preceding_scene = select_preceding_context(
                generated_latents=generated_latents,
                generated_scene_latents=generated_scene_latents,
                num_frames=preceding_latent_frames,
            )
            reference_latents, reference_indices = select_reference_latents(
                lpc=lpc,
                geometry=geometry,
                target_pose_indices=target_pose_indices,
                generated_latents=generated_latents,
                frame_visible_points=frame_visible_points,
                max_reference_frames=self.config.max_reference_frames,
                iou_threshold=self.config.ref_iou_threshold,
                voxel_size=self.config.ref_iou_voxel_size,
                exclude_newest=1 + preceding_latent_frames
                if self.config.exclude_recent_refs
                else 0,
            )
            if (
                iter_idx > 0
                and self.config.reencode_references
                and (reference_latents is not None)
                and reference_indices
            ):
                self.pipe.load_models_to_device(["vae"])
                hist_latents = torch.stack(generated_latents, dim=1).unsqueeze(0)
                hist_px = decode_latents_to_uint8(
                    self.pipe,
                    hist_latents,
                    tiled=self.config.tiled,
                    tile_size=self.config.tile_size,
                    tile_stride=self.config.tile_stride,
                )
                ref_encs = []
                for lat_idx in reference_indices:
                    frame_idx = latent_to_frame.get(int(lat_idx))
                    if frame_idx is None or not 0 <= frame_idx < len(hist_px):
                        raise ValueError(
                            f"reencode_references: latent {lat_idx} maps to frame {frame_idx}, outside the {len(hist_px)}-frame joint history decode."
                        )
                    ref_encs.append(
                        encode_video_frames(
                            self.pipe,
                            hist_px[frame_idx][None],
                            tiled=self.config.tiled,
                            tile_size=self.config.tile_size,
                            tile_stride=self.config.tile_stride,
                        ).to(device=device, dtype=dtype)
                    )
                re_refs = torch.cat(ref_encs, dim=2)
                if re_refs.shape[2] != reference_latents.shape[2]:
                    raise ValueError(
                        f"reencode_references produced {re_refs.shape[2]} latents but selection returned {reference_latents.shape[2]}."
                    )
                reference_latents = re_refs
            boundary_anchor_latent = None
            if iter_idx > 0 and self.config.reencode_anchor:
                self.pipe.load_models_to_device(["vae"])
                boundary_anchor_latent = encode_video_frames(
                    self.pipe,
                    np.asarray(generated_frames[-1])[None],
                    tiled=self.config.tiled,
                    tile_size=self.config.tile_size,
                    tile_stride=self.config.tile_stride,
                ).to(device=device, dtype=dtype)
                if boundary_anchor_latent.shape[2] != 1:
                    raise ValueError(
                        f"reencode_anchor: expected 1 latent, got {boundary_anchor_latent.shape[2]}"
                    )
            if (
                iter_idx > 0
                and self.config.reencode_preceding
                and (preceding_latents is not None)
            ):
                self.pipe.load_models_to_device(["vae"])
                npx = self.config.preceding_pixel_frames
                if len(generated_frames) < npx + 1:
                    raise ValueError(
                        f"reencode_preceding needs {npx + 1} generated frames, have {len(generated_frames)}."
                    )
                hist_px = np.stack(generated_frames[-(npx + 1) : -1])
                re_p = encode_video_frames(
                    self.pipe,
                    hist_px,
                    tiled=self.config.tiled,
                    tile_size=self.config.tile_size,
                    tile_stride=self.config.tile_stride,
                ).to(device=device, dtype=dtype)
                if re_p.shape[2] != preceding_latents.shape[2]:
                    raise ValueError(
                        f"reencode_preceding produced {re_p.shape[2]} latents but the preceding window has {preceding_latents.shape[2]}."
                    )
                preceding_latents = re_p
            if iter_idx == 0:
                iter_first_latent = first_frame_latent
            elif boundary_anchor_latent is not None:
                iter_first_latent = boundary_anchor_latent
            else:
                iter_first_latent = generated_latents[-1][None, :, None]
            self._vace_scale_current = 1.0 if iter_idx == 0 else float("1.0")
            output_latents = self.generate_single_iteration(
                target_scene=target_scene,
                first_frame_latent=iter_first_latent,
                preceding_latents=preceding_latents,
                preceding_scene=preceding_scene,
                reference_latents=reference_latents,
                context=context,
                uncon_context=uncon_context,
                generator=generator,
            )
            if generated_latents:
                _joint = torch.cat(
                    [
                        torch.stack(generated_latents, dim=1).unsqueeze(0),
                        output_latents[:, :, 1:],
                    ],
                    dim=2,
                )
                _joint_px = decode_latents_to_uint8(
                    self.pipe,
                    _joint,
                    tiled=self.config.tiled,
                    tile_size=self.config.tile_size,
                    tile_stride=self.config.tile_stride,
                )
                _n_chunk = 1 + temporal_stride * (int(output_latents.shape[2]) - 1)
                iter_video = _joint_px[-_n_chunk:]
            else:
                iter_video = decode_latents_to_uint8(
                    self.pipe,
                    output_latents,
                    tiled=self.config.tiled,
                    tile_size=self.config.tile_size,
                    tile_stride=self.config.tile_stride,
                )
            iter_video_path = (
                video_dir
                / f"iteration_{iter_idx + 1:02d}_frames{output_start}-{output_end - 1}.mp4"
            )
            write_iteration_video(
                iter_video_path, iter_video, iter_idx, self.config.fps
            )
            output_latents_tchw = rearrange(output_latents[0], "c t h w -> t c h w")
            target_scene_tchw = rearrange(target_scene, "c t h w -> t c h w").to(
                device=output_latents_tchw.device, dtype=output_latents_tchw.dtype
            )
            if iter_idx == 0:
                new_latents = output_latents_tchw
                new_scene = target_scene_tchw
                new_pose_indices = target_pose_indices
                new_images = select_latent_aligned_frames(iter_video, temporal_stride)
                generated_frames.extend(list(iter_video))
            else:
                new_latents = output_latents_tchw[1:]
                new_scene = target_scene_tchw[1:]
                new_pose_indices = target_pose_indices[1:]
                new_images = select_latent_aligned_frames(iter_video, temporal_stride)[
                    1:
                ]
                generated_frames.extend(list(iter_video[1:]))
            generated_latents.extend(list(torch.unbind(new_latents.detach(), dim=0)))
            generated_scene_latents.extend(
                list(torch.unbind(new_scene.detach(), dim=0))
            )
            if self.config.update_memory:
                mem_images = new_images
                mem_pose_indices = new_pose_indices
                mem_latents = new_latents
                mem_times = list(
                    range(
                        len(generated_latents) - len(new_pose_indices),
                        len(generated_latents),
                    )
                )
                self.update_latent_memory(
                    lpc=lpc,
                    geometry=geometry,
                    images=mem_images,
                    pose_indices=mem_pose_indices,
                    latents=mem_latents,
                    output_latent_times=mem_times,
                )
                update_frame_visibility(
                    lpc=lpc,
                    geometry=geometry,
                    pose_indices=new_pose_indices,
                    frame_visible_points=frame_visible_points,
                    start_output_latent=len(generated_latents) - len(new_pose_indices),
                    latent_to_frame=latent_to_frame,
                    start_frame=self.config.start_frame,
                )
            metadata["iterations"].append(
                {
                    "iteration": iter_idx + 1,
                    "output_start": output_start,
                    "output_end": output_end,
                    "model_frames": model_frames,
                    "target_pose_indices": target_pose_indices,
                    "reference_indices": reference_indices,
                    "num_preceding_latents": 0
                    if preceding_latents is None
                    else int(preceding_latents.shape[2]),
                    "num_reference_latents": 0
                    if reference_latents is None
                    else int(reference_latents.shape[2]),
                }
            )
        per_chunk_video = np.stack(generated_frames, axis=0)[:requested_frames]
        if generated_latents:
            joint_latents = torch.stack(generated_latents, dim=1).unsqueeze(0)
            final_video = decode_latents_to_uint8(
                self.pipe,
                joint_latents,
                tiled=self.config.tiled,
                tile_size=self.config.tile_size,
                tile_stride=self.config.tile_stride,
            )[:requested_frames]
            metadata["joint_decode_latents"] = int(joint_latents.shape[2])
            metadata["generated_mp4_decode"] = "joint"
        else:
            final_video = per_chunk_video
            metadata["generated_mp4_decode"] = "per_chunk_fallback"
        write_mp4(video_dir / "generated.mp4", final_video, self.config.fps)
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        return torch.from_numpy(final_video)

    def encode_prompt(self, prompt: str, *, positive: bool) -> Tensor:
        self.pipe.load_models_to_device(["text_encoder"])
        return self.pipe.prompter.encode_prompt(
            prompt, positive=positive, device=self.pipe.device
        ).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)

    def generate_single_iteration(
        self,
        *,
        target_scene: Tensor,
        first_frame_latent: Tensor,
        preceding_latents: Tensor | None,
        preceding_scene: Tensor | None,
        reference_latents: Tensor | None,
        context: Tensor,
        uncon_context: Tensor | None,
        generator: torch.Generator,
    ) -> Tensor:
        device = torch.device(self.pipe.device)
        dtype = self.pipe.torch_dtype
        channels, num_t, latent_h, latent_w = target_scene.shape
        num_p = 0 if preceding_latents is None else preceding_latents.shape[2]
        num_r = 0 if reference_latents is None else reference_latents.shape[2]
        noise = torch.randn(
            (1, channels, num_t, latent_h, latent_w),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents_t = noise
        latents_t[:, :, :1].copy_(first_frame_latent.to(device=device, dtype=dtype))
        vace_context = build_vace_context(
            target_scene=target_scene,
            preceding_scene=preceding_scene,
            dtype=dtype,
            device=device,
            preceding_first=self.config.preceding_first,
        )
        self.pipe.load_models_to_device(["dit", "vace"])
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=1.0,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            self.config.infer_steps, device=device, shift=self.config.timestep_shift
        )
        timestep_p = (
            torch.zeros(num_p, device=device, dtype=torch.float32)
            if num_p > 0
            else None
        )
        timestep_r = (
            torch.zeros(num_r, device=device, dtype=torch.float32)
            if num_r > 0
            else None
        )
        for step_idx, timestep in enumerate(
            tqdm(
                scheduler.timesteps, desc=f"Denoising (T={num_t}, P={num_p}, R={num_r})"
            )
        ):
            timestep_t = timestep.to(device=device, dtype=torch.float32).repeat(num_t)
            timestep_t[0] = 0
            pre_first = self.config.preceding_first
            timestep_parts = []
            if timestep_r is not None:
                timestep_parts.append(timestep_r)
            if pre_first and timestep_p is not None:
                timestep_parts.append(timestep_p)
            timestep_parts.append(timestep_t)
            if not pre_first and timestep_p is not None:
                timestep_parts.append(timestep_p)
            full_timestep = torch.cat(timestep_parts, dim=0)
            latent_parts = []
            if reference_latents is not None and num_r > 0:
                latent_parts.append(reference_latents.to(device=device, dtype=dtype))
            have_p = preceding_latents is not None and num_p > 0
            if pre_first and have_p:
                latent_parts.append(preceding_latents.to(device=device, dtype=dtype))
            latent_parts.append(latents_t)
            if not pre_first and have_p:
                latent_parts.append(preceding_latents.to(device=device, dtype=dtype))
            combined_latents = torch.cat(latent_parts, dim=2)
            target_offset = num_r + (num_p if pre_first else 0)
            model_kwargs = {
                "dit": self.pipe.dit,
                "vace": self.pipe.vace,
                "latents": combined_latents,
                "timestep": full_timestep,
                "context": context,
                "vace_context": vace_context,
                "vace_scale": float(getattr(self, "_vace_scale_current", 1.0)),
                "num_ref_frames": num_r,
                "fuse_vae_embedding_in_latents": True,
            }
            if step_idx == 0:
                print(
                    f"latents={tuple(combined_latents.shape)} vace_context={(1, *tuple(vace_context[0].shape))} order={('R|P|T' if pre_first else 'R|T|P')} vace_scale={model_kwargs['vace_scale']} first_timestep={float(full_timestep[target_offset].item())}"
                )
            flow_cond = model_fn_wan_video(**model_kwargs)
            if not self.config.no_cfg and uncon_context is not None:
                model_kwargs["context"] = uncon_context
                flow_uncond = model_fn_wan_video(**model_kwargs)
                flow = flow_uncond + self.config.guidance_scale * (
                    flow_cond - flow_uncond
                )
            else:
                flow = flow_cond
            flow_t = flow[:, :, target_offset : target_offset + num_t]
            latents_t = scheduler.step(
                flow_t, timestep, latents_t, return_dict=False, generator=generator
            )[0]
            latents_t[:, :, :1].copy_(first_frame_latent.to(device=device, dtype=dtype))
        return latents_t

    def update_latent_memory(
        self,
        *,
        lpc: LatentPointCloud,
        geometry: VideoGeometry,
        images: np.ndarray,
        pose_indices: list[int],
        latents: Tensor,
        output_latent_times: list[int] | None = None,
    ) -> None:
        if len(images) == 0:
            return
        predictions = infer_da3_depths(
            images=images,
            geometry=geometry,
            pose_indices=pose_indices,
            python_executable=self.config.depth_python,
        )
        depths = np.stack([item["depth"] for item in predictions], axis=0)
        depth_hw = depths.shape[1:3]
        intrinsics = np.stack(
            [
                scale_intrinsics_to_hw(
                    geometry.intrinsics[pose_idx], geometry.frames.shape[1:3], depth_hw
                )
                for pose_idx in pose_indices
            ],
            axis=0,
        )
        poses = geometry.poses_c2w[np.asarray(pose_indices)]
        if isinstance(lpc, LatentPointCloud):
            lpc.update(
                depths=depths, intrinsics=intrinsics, cam2worlds=poses, latents=latents
            )
            return
        if output_latent_times is None:
            raise ValueError(
                "output_latent_times is required for a memory_factory backend"
            )
        if len(output_latent_times) != int(latents.shape[0]):
            raise ValueError(
                f"{len(output_latent_times)} output_latent_times for {int(latents.shape[0])} memory latents; expected one time per latent."
            )
        keep = [i for i, t in enumerate(output_latent_times) if int(t) != 0]
        if not keep:
            return
        lpc.update(
            depths=depths[keep],
            intrinsics=intrinsics[keep],
            cam2worlds=poses[keep],
            latents=latents[keep],
            times=[int(output_latent_times[i]) for i in keep],
        )


def load_video_geometry_for_inference(path: Path, *, start_frame: int) -> VideoGeometry:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        frames = np.asarray(data["frames"]) if "frames" in data.files else None
        depths = np.asarray(data["depths"], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics"])
        poses_c2w = np.asarray(data["poses_c2w"])
        masks = load_optional_array(data, "masks")
        frame_indices = load_optional_array(data, "frame_indices")
        original_size = load_optional_hw(data, "original_size")
        processed_size = load_optional_hw(data, "processed_size")
    if frames is None:
        frames = load_geometry_rgb_frames(
            sample_dir=path.parent,
            processed_size=processed_size,
            start_frame=start_frame,
        )
    return VideoGeometry(
        frames=frames,
        depths=depths,
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
        masks=masks,
        frame_indices=frame_indices,
        original_size=original_size,
        processed_size=processed_size,
    )


def load_optional_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.size == 0:
        return None
    return value


def load_optional_hw(data: np.lib.npyio.NpzFile, key: str) -> tuple[int, int] | None:
    value = load_optional_array(data, key)
    if value is None:
        return None
    hw = tuple((int(x) for x in value.tolist()))
    if len(hw) != 2 or hw == (-1, -1):
        return None
    return hw


def load_geometry_rgb_frames(
    *, sample_dir: Path, processed_size: tuple[int, int] | None, start_frame: int
) -> np.ndarray:
    clip_path = sample_dir / "clip.mp4"
    if clip_path.exists():
        return load_video_rgb_frames(clip_path, target_hw=processed_size)
    first_frame_path = sample_dir / "first_frame.png"
    if first_frame_path.exists():
        if start_frame != 0:
            raise ValueError(
                f"geometry.npz has no frames and only first_frame.png is available; cannot recover start_frame={start_frame}."
            )
        image = Image.open(first_frame_path).convert("RGB")
        if processed_size is not None:
            image = image.resize(
                (processed_size[1], processed_size[0]), Image.Resampling.BILINEAR
            )
        return np.asarray(image, dtype=np.uint8)[None]
    raise FileNotFoundError(
        f"geometry.npz has no frames. Expected clip.mp4 or first_frame.png in {sample_dir}."
    )


def load_video_rgb_frames(
    path: Path, *, target_hw: tuple[int, int] | None
) -> np.ndarray:
    import imageio.v2 as imageio

    frames = []
    reader = None
    try:
        reader = imageio.get_reader(str(path), format="FFMPEG")
        for frame in reader:
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            if target_hw is not None:
                image = image.resize(
                    (target_hw[1], target_hw[0]), Image.Resampling.BILINEAR
                )
            frames.append(np.asarray(image, dtype=np.uint8))
    finally:
        if reader is not None:
            reader.close()
    if not frames:
        raise RuntimeError(f"No frames loaded from video: {path}")
    return np.stack(frames, axis=0)


def pad_geometry_to(geometry: VideoGeometry, length: int) -> VideoGeometry:
    def extend(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None or len(arr) >= length:
            return arr
        pad = np.repeat(arr[-1:], length - len(arr), axis=0)
        return np.concatenate([arr, pad], axis=0)

    return VideoGeometry(
        frames=extend(geometry.frames) if len(geometry.frames) > 1 else geometry.frames,
        depths=extend(geometry.depths),
        intrinsics=extend(geometry.intrinsics),
        poses_c2w=extend(geometry.poses_c2w),
        masks=extend(geometry.masks),
        frame_indices=extend(geometry.frame_indices),
        original_size=geometry.original_size,
        processed_size=geometry.processed_size,
    )


def full_chunk_length(num_frames: int) -> int:
    if num_frames <= MAX_CHUNK:
        return num_frames
    k = -(-(num_frames - MAX_CHUNK) // (MAX_CHUNK - 1))
    return MAX_CHUNK + k * (MAX_CHUNK - 1)


def validate_geometry(geometry: VideoGeometry, *, start_frame: int) -> None:
    required = {
        "frames": geometry.frames,
        "depths": geometry.depths,
        "intrinsics": geometry.intrinsics,
        "poses_c2w": geometry.poses_c2w,
    }
    for name, value in required.items():
        if value is None:
            raise ValueError(f"geometry.npz is missing {name}.")
    if len(geometry.depths) != len(geometry.poses_c2w):
        raise ValueError("geometry depths/poses_c2w length mismatch.")
    if len(geometry.intrinsics) != len(geometry.poses_c2w):
        raise ValueError("geometry intrinsics/poses_c2w length mismatch.")
    if len(geometry.frames) not in {1, len(geometry.poses_c2w)}:
        raise ValueError(
            "geometry RGB frames must contain either the first frame or the full pose sequence."
        )
    if start_frame >= len(geometry.frames):
        raise ValueError(
            f"RGB frames contain {len(geometry.frames)} frame(s), cannot read start_frame={start_frame}."
        )


def resolve_output_hw(
    geometry: VideoGeometry, config: InferenceConfig
) -> tuple[int, int]:
    if config.height is not None or config.width is not None:
        if config.height is None or config.width is None:
            raise ValueError("--height and --width must be provided together.")
        return (int(config.height), int(config.width))
    if geometry.original_size is not None:
        return (int(geometry.original_size[0]), int(geometry.original_size[1]))
    return (int(geometry.frames.shape[1]), int(geometry.frames.shape[2]))


def get_initial_exclusion_mask(
    geometry: VideoGeometry, frame_idx: int
) -> np.ndarray | None:
    if geometry.masks is None:
        return None
    return ~geometry.masks[frame_idx].astype(bool)


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    image = Image.fromarray(frame.astype(np.uint8))
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))


def encode_video_frames(
    pipe: WanVideoPipeline,
    frames: np.ndarray,
    *,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> Tensor:
    vae_dtype = next(pipe.vae.parameters()).dtype
    video = torch.from_numpy(frames).float()
    video = rearrange(video, "t h w c -> c t h w").div(127.5).sub(1.0)
    video = video.to(dtype=vae_dtype)
    return pipe.vae.encode(
        [video],
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


def decode_latents_to_uint8(
    pipe: WanVideoPipeline,
    latents: Tensor,
    *,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> np.ndarray:
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    video = video[0].add(1.0).mul(0.5).clamp(0.0, 1.0)
    video = rearrange(video, "c t h w -> t h w c")
    return video.mul(255).byte().cpu().numpy()


def build_target_pose_indices(
    *,
    start_frame: int,
    output_start: int,
    model_frames: int,
    temporal_stride: int,
    iter_idx: int,
) -> list[int]:
    if iter_idx == 0:
        pose_start = start_frame + output_start
    else:
        pose_start = start_frame + output_start - 1
    num_latent_frames = (model_frames - 1) // temporal_stride + 1
    return [pose_start + i * temporal_stride for i in range(num_latent_frames)]


def project_lpc_sequence(
    *, lpc: LatentPointCloud, geometry: VideoGeometry, frame_indices: list[int]
) -> Tensor:
    latents = []
    for frame_idx in frame_indices:
        intrinsics_latent = scale_intrinsics_to_hw(
            geometry.intrinsics[frame_idx], geometry.frames.shape[1:3], lpc.latent_hw
        )
        latent, _ = lpc.project(
            cam2world=geometry.poses_c2w[frame_idx], intrinsics=intrinsics_latent
        )
        latents.append(latent)
    return torch.stack(latents, dim=1)


def select_preceding_context(
    *,
    generated_latents: list[Tensor],
    generated_scene_latents: list[Tensor],
    num_frames: int,
) -> tuple[Tensor | None, Tensor | None]:
    if len(generated_latents) <= 1 or num_frames <= 0:
        return (None, None)
    stop = len(generated_latents) - 1
    start = max(0, stop - num_frames)
    if start == stop:
        return (None, None)
    latents = torch.stack(generated_latents[start:stop], dim=1).unsqueeze(0)
    scene = torch.stack(generated_scene_latents[start:stop], dim=1)
    return (latents, scene)


def select_reference_latents(
    *,
    lpc: LatentPointCloud,
    geometry: VideoGeometry,
    target_pose_indices: list[int],
    generated_latents: list[Tensor],
    frame_visible_points: dict[int, np.ndarray],
    max_reference_frames: int,
    iou_threshold: float,
    voxel_size: float,
    exclude_newest: int = 0,
) -> tuple[Tensor | None, list[int]]:
    if exclude_newest > 0:
        cutoff = len(generated_latents) - exclude_newest
        frame_visible_points = {
            idx: points for idx, points in frame_visible_points.items() if idx < cutoff
        }
    if max_reference_frames <= 0 or not generated_latents or (not frame_visible_points):
        return (None, [])
    target_points = []
    for frame_idx in target_pose_indices:
        target_points.append(visible_points_from_lpc(lpc, geometry, frame_idx))
    target_points = [points for points in target_points if len(points) > 0]
    if not target_points:
        return (None, [])
    target_points_combined = np.concatenate(target_points, axis=0)
    scored = []
    for hist_idx, hist_points in frame_visible_points.items():
        iou = compute_points_iou(target_points_combined, hist_points, voxel_size)
        if iou >= iou_threshold:
            scored.append((hist_idx, iou))
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = [idx for idx, _ in scored[:max_reference_frames]]
    if not selected:
        return (None, [])
    ref_latents = torch.stack([generated_latents[idx] for idx in selected], dim=1)
    return (ref_latents.unsqueeze(0), selected)


def visible_points_from_lpc(
    lpc: LatentPointCloud, geometry: VideoGeometry, frame_idx: int
) -> np.ndarray:
    device = lpc.points_world.device
    points_world = lpc.points_world[lpc.valid_mask.bool()]
    cam2world = torch.as_tensor(
        geometry.poses_c2w[frame_idx], device=device, dtype=torch.float32
    )
    intrinsics = torch.as_tensor(
        scale_intrinsics_to_hw(
            geometry.intrinsics[frame_idx], geometry.frames.shape[1:3], lpc.latent_hw
        ),
        device=device,
        dtype=torch.float32,
    )
    world2cam = torch.inverse(cam2world)
    points_cam = points_world @ world2cam[:3, :3].T + world2cam[:3, 3]
    z = points_cam[:, 2]
    u = points_cam[:, 0] * intrinsics[0, 0] / z + intrinsics[0, 2]
    v = points_cam[:, 1] * intrinsics[1, 1] / z + intrinsics[1, 2]
    height, width = lpc.latent_hw
    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return points_world[valid].detach().cpu().numpy().astype(np.float32)


def compute_points_iou(
    points_a: np.ndarray, points_b: np.ndarray, voxel_size: float
) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    vox_a = np.floor(points_a / voxel_size).astype(np.int32)
    vox_b = np.floor(points_b / voxel_size).astype(np.int32)
    set_a = set(map(tuple, vox_a.tolist()))
    set_b = set(map(tuple, vox_b.tolist()))
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def update_frame_visibility(
    *,
    lpc: LatentPointCloud,
    geometry: VideoGeometry,
    pose_indices: list[int],
    frame_visible_points: dict[int, np.ndarray],
    start_output_latent: int,
    latent_to_frame: dict[int, int] | None = None,
    start_frame: int = 0,
) -> None:
    for offset, pose_idx in enumerate(pose_indices):
        frame_visible_points[start_output_latent + offset] = visible_points_from_lpc(
            lpc, geometry, pose_idx
        )
        if latent_to_frame is not None:
            latent_to_frame[start_output_latent + offset] = int(pose_idx) - start_frame


def build_vace_context(
    *, target_scene, preceding_scene, dtype, device, preceding_first=True
):
    if preceding_scene is None:
        scene = target_scene
    else:
        scene = torch.cat(
            [preceding_scene, target_scene]
            if preceding_first
            else [target_scene, preceding_scene],
            dim=1,
        )
    if scene.shape[0] != 48:
        raise ValueError(f"Expected 48-channel scene latent, got {scene.shape[0]}.")
    return [
        append_visibility_channel(scene.to(device=device, dtype=dtype), channel_dim=0)
    ]


def infer_da3_depths(
    *,
    images: np.ndarray,
    geometry: VideoGeometry,
    pose_indices: list[int],
    python_executable: str,
) -> list[dict[str, np.ndarray]]:
    import subprocess, tempfile

    py = python_executable
    worker = str(Path(__file__).with_name("depth_worker.py"))
    image_hw = images.shape[1:3]
    fx = np.array(
        [
            scale_intrinsics_to_hw(
                geometry.intrinsics[i], geometry.frames.shape[1:3], image_hw
            )[0, 0]
            for i in pose_indices
        ],
        dtype=np.float32,
    )
    with tempfile.TemporaryDirectory(prefix="da3_") as td:
        fin, fout = (os.path.join(td, "in.npz"), os.path.join(td, "out.npz"))
        np.savez(fin, images=np.asarray(images, dtype=np.uint8), fx=fx)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "PYTHONNOUSERSITE")
        }
        subprocess.run([py, worker, fin, fout], check=True, env=env)
        depths = np.load(fout)["depths"]
    if depths.shape[0] != len(pose_indices) or depths.shape[1:] != tuple(image_hw):
        raise ValueError(
            f"da3 worker returned {depths.shape}, expected ({len(pose_indices)}, {image_hw})"
        )
    return [{"depth": d.astype(np.float32)} for d in depths]


def scale_intrinsics_to_hw(
    intrinsics: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scaled = intrinsics.copy().astype(np.float32)
    scaled[0, 0] *= target_w / source_w
    scaled[0, 2] *= target_w / source_w
    scaled[1, 1] *= target_h / source_h
    scaled[1, 2] *= target_h / source_h
    return scaled


def select_latent_aligned_frames(video: np.ndarray, temporal_stride: int) -> np.ndarray:
    return video[::temporal_stride]


def write_iteration_video(
    path: Path, video: np.ndarray, iter_idx: int, fps: int
) -> None:
    frames = video if iter_idx == 0 else video[1:]
    write_mp4(path, frames, fps)


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    video = frames
    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    writer = None
    try:
        writer = imageio.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        for frame in video:
            writer.append_data(frame)
    finally:
        if writer is not None:
            writer.close()


def load_lora_checkpoint(
    pipe: WanVideoPipeline, path: Path, *, alpha: float, fuse: bool = True
) -> dict:
    state = load_state_dict(str(path), torch_dtype=pipe.torch_dtype, device=pipe.device)
    state = _normalize_lora_state_dict(state)
    if fuse:
        pipe.load_lora(pipe.dit, lora_state_dict=state, alpha=alpha)
    return state


def _normalize_lora_state_dict(state: dict[str, Any]) -> dict[str, Tensor]:
    normalized_state: dict[str, Tensor] = {}
    source_keys: dict[str, str] = {}
    for key, value in state.items():
        if not torch.is_tensor(value) or "lora_" not in key:
            continue
        normalized_key = _normalize_lora_checkpoint_key(key)
        if normalized_key in normalized_state:
            raise ValueError(
                f"LoRA checkpoint contains duplicate tensors after normalization: '{source_keys[normalized_key]}' and '{key}' both map to '{normalized_key}'."
            )
        normalized_state[normalized_key] = value
        source_keys[normalized_key] = key
    if not normalized_state:
        raise ValueError("No LoRA tensors found in LoRA checkpoint.")
    return normalized_state


def _normalize_lora_checkpoint_key(key: str) -> str:
    normalized = strip_prefixes(
        key,
        ("module.", "model.", "pipe.dit.", "dit.", "base_model.model.", "base_model."),
    )
    normalized = normalized.replace(".lora_A.default.weight", ".lora_A.weight")
    normalized = normalized.replace(".lora_B.default.weight", ".lora_B.weight")
    return normalized
