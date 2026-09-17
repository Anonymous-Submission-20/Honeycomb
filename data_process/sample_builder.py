from typing import Optional

import cv2
import numpy as np
import torch

from data_process.data_config import SampleConfig
from data_process.reference_frames import (
    RefSelectionResult,
    select_reference_frames,
)
from data_process.sample_indices import (
    filter_reference_candidates,
    sample_frame_indices,
)
from data_process.types import SampleIndices, VideoGeometry
from lsm.latent_point_cloud import LatentPointCloud


def resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    resized = []
    for frame in frames:
        resized.append(
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        )
    return np.stack(resized, axis=0)


def scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    return K_scaled


def build_scene_exclusion_mask(
    geometry: VideoGeometry,
    dynamic_masks: Optional[np.ndarray],
    scene_idx: int,
) -> Optional[np.ndarray]:
    exclusion_mask = None
    if geometry.masks is not None:
        exclusion_mask = ~geometry.masks[scene_idx]
    if dynamic_masks is not None:
        dynamic_mask = dynamic_masks[scene_idx]
        exclusion_mask = (
            dynamic_mask.copy()
            if exclusion_mask is None
            else (exclusion_mask | dynamic_mask)
        )
    return exclusion_mask


def build_latent_projection(
    latent_point_cloud: LatentPointCloud,
    geometry: VideoGeometry,
    frame_indices: list[int],
    temporal_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    sampled_frame_indices = list(frame_indices)[::temporal_stride]
    projections, masks = _project_latent_frames_to_numpy(
        latent_point_cloud=latent_point_cloud,
        geometry=geometry,
        frame_indices=sampled_frame_indices,
    )
    return projections, masks, sampled_frame_indices


def _project_latent_frames_to_numpy(
    latent_point_cloud: LatentPointCloud,
    geometry: VideoGeometry,
    frame_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not frame_indices:
        raise ValueError("frame_indices must contain at least one frame")

    H, W = geometry.frames.shape[1:3]
    latent_h, latent_w = latent_point_cloud.latent_hw
    scale_x = latent_w / W
    scale_y = latent_h / H

    projections = []
    masks = []
    for idx in frame_indices:
        intrinsics_latent = scale_intrinsics(geometry.intrinsics[idx], scale_x, scale_y)
        proj, proj_mask = latent_point_cloud.project(
            cam2world=geometry.poses_c2w[idx],
            intrinsics=intrinsics_latent,
        )
        projections.append(proj)
        masks.append(proj_mask)

    projection_array = torch.stack(projections, dim=0).detach().cpu().numpy()
    projection_array = projection_array.astype(np.float32, copy=False)
    mask_array = torch.stack(masks, dim=0).detach().cpu().numpy()
    return projection_array, mask_array


def build_training_sample(
    geometry: VideoGeometry,
    config: SampleConfig,
    dynamic_masks: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    original_frames: Optional[np.ndarray] = None,
    output_size: Optional[tuple[int, int]] = None,
    indices: Optional[SampleIndices] = None,
    scene_latent: Optional[torch.Tensor] = None,
    latent_projection_stride: int = 4,
):
    assert scene_latent is not None, (
        "scene_latent must be provided for latent scene projection"
    )

    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    if indices is None:
        num_frames = geometry.frames.shape[0]
        indices = sample_frame_indices(
            num_frames=num_frames,
            N_target=config.N_target,
            M_pre=config.M_pre,
            min_gap_for_candidates=config.min_gap_for_candidates,
            rng=rng,
        )

    scene_idx = int(indices.t0)

    H, W = geometry.frames.shape[1:3]

    preceding_proj_mask = None
    target_proj_mask = None
    apply_masks = getattr(config, "apply_dynamic_mask", True)
    dyn_masks = dynamic_masks if apply_masks else None
    exclusion_mask = (
        build_scene_exclusion_mask(
            geometry=geometry,
            dynamic_masks=dyn_masks,
            scene_idx=scene_idx,
        )
        if apply_masks
        else None
    )
    latent_point_cloud = LatentPointCloud.from_video_geometry(
        geometry=geometry,
        frame_idx=scene_idx,
        latent=scene_latent,
        mask=exclusion_mask,
        device=scene_latent.device,
    )
    valid_scene_points = (
        latent_point_cloud.points_world[latent_point_cloud.valid_mask]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    assert valid_scene_points.size > 0, (
        "Latent scene projection has no valid points; check geometry."
    )
    preceding_proj_indices = list(indices.preceding_indices)[
        ::latent_projection_stride
    ]
    target_proj_indices = list(indices.target_indices)[::latent_projection_stride]
    all_projection_indices = preceding_proj_indices + target_proj_indices
    all_projections, all_projection_masks = _project_latent_frames_to_numpy(
        latent_point_cloud=latent_point_cloud,
        geometry=geometry,
        frame_indices=all_projection_indices,
    )
    preceding_count = len(preceding_proj_indices)
    preceding_proj = all_projections[:preceding_count]
    target_proj = all_projections[preceding_count:]
    preceding_proj_mask = all_projection_masks[:preceding_count]
    target_proj_mask = all_projection_masks[preceding_count:]

    # use coarse voxels for reference selection to tolerate depth noise
    iou_voxel_size = config.ref_iou_voxel_size
    if iou_voxel_size is None:
        iou_voxel_size = config.scene_voxel_size * 10

    ref_candidates = filter_reference_candidates(
        list(indices.candidate_indices),
        list(indices.preceding_indices),
        config.ref_candidate_scope,
    )
    ref_result: RefSelectionResult = select_reference_frames(
        candidate_indices=ref_candidates,
        target_indices=indices.target_indices,
        depths=geometry.depths,
        intrinsics=geometry.intrinsics,
        poses_c2w=geometry.poses_c2w,
        voxel_size=iou_voxel_size,
        stride=config.K_ref_stride,
        iou_threshold=config.eps_iou,
        max_refs=config.max_refs,
        # use the same exclusion mask for reference selection and memory construction
        valid_masks=geometry.masks if apply_masks else None,
        dynamic_masks=dyn_masks,
        return_result=True,
    )
    reference_indices = ref_result.indices
    reference_ious = ref_result.ious

    if original_frames is not None:
        src_frames = np.asarray(original_frames)
    else:
        src_frames = geometry.frames

    if output_size is not None:
        out_h, out_w = output_size
    elif geometry.original_size is not None:
        out_h, out_w = geometry.original_size
    else:
        out_h, out_w = H, W

    preceding_rgb = src_frames[indices.preceding_indices]
    target_rgb = src_frames[indices.target_indices]
    if reference_indices:
        reference_rgb = src_frames[reference_indices]
    else:
        reference_rgb = np.zeros((0, out_h, out_w, 3), dtype=np.uint8)

    preceding_rgb = resize_frames(preceding_rgb, (out_h, out_w))
    target_rgb = resize_frames(target_rgb, (out_h, out_w))
    if reference_rgb.shape[0] > 0:
        reference_rgb = resize_frames(reference_rgb, (out_h, out_w))

    sample = {
        "P_rgb": preceding_rgb,
        "T_rgb": target_rgb,
        "R_rgb": reference_rgb,
        "P_poses_c2w": geometry.poses_c2w[indices.preceding_indices],
        "T_poses_c2w": geometry.poses_c2w[indices.target_indices],
        "P_intrinsics": geometry.intrinsics[indices.preceding_indices],
        "T_intrinsics": geometry.intrinsics[indices.target_indices],
        "proj_P": preceding_proj,
        "proj_T": target_proj,
        "meta": {
            "t0": indices.t0,
            "P_idx": indices.preceding_indices,
            "T_idx": indices.target_indices,
            "C_idx": indices.candidate_indices,
            "scene_idx": scene_idx,
            "R_idx": reference_indices,
            "R_iou": reference_ious,
            "R_stats": ref_result.stats,
            "ref_candidate_scope": config.ref_candidate_scope,
            "point_cloud_type": "latent",
            "projection_channels": ["latent"],
            "projection_layout": "tchw",
            "latent_projection_stride": latent_projection_stride,
            "proj_P_idx": preceding_proj_indices,
            "proj_T_idx": target_proj_indices,
            "output_size": (out_h, out_w),
        },
    }
    if preceding_proj_mask is not None:
        sample["proj_P_mask"] = preceding_proj_mask
    if target_proj_mask is not None:
        sample["proj_T_mask"] = target_proj_mask
    return sample
