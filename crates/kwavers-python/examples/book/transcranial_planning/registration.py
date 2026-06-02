from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform

from .data import BrainTriplet, Volume, normalize_unit
from .metrics import (
    binary_edge_overlap,
    dice_overlap,
    foreground_mask,
    multimodal_affine_score,
    normalized_cross_correlation,
    normalized_mean_squared_error,
    normalized_mutual_information,
    registration_quality,
)


@dataclass(frozen=True)
class RegistrationResult:
    ct: Volume
    t1_registered: Volume
    atlas_registered: Volume
    executed: bool
    method: str
    ncc_t1_before: float
    ncc_t1_after: float
    nmi_t1_before: float
    nmi_t1_after: float
    mse_t1_before: float
    mse_t1_after: float
    ncc_atlas_before: float
    ncc_atlas_after: float
    nmi_atlas_before: float
    nmi_atlas_after: float
    mse_atlas_before: float
    mse_atlas_after: float
    message: str


@dataclass(frozen=True)
class AffineRegistrationResult:
    fixed: Volume
    moving_registered: Volume
    executed: bool
    method: str
    nmi: float
    edge_overlap: float
    message: str


def affine_register_moving_to_fixed(
    moving: Volume,
    fixed: Volume,
    name: str,
    order: int = 1,
    strategy: str = "world",
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    axis_reflections: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] = (
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ),
) -> AffineRegistrationResult:
    """Resample a NIfTI moving volume onto a fixed volume using world affines.

    For an output voxel index `i_f`, the fixed affine maps to world coordinates
    and the inverse moving affine maps that point to moving index `i_m`.
    `scipy.ndimage.affine_transform` therefore receives the matrix in
    output-index to input-index form:

    `i_m = inv(A_moving) @ A_fixed @ i_f`.
    """

    if strategy == "world":
        transform = np.linalg.inv(moving.affine) @ fixed.affine
        matrix = transform[:3, :3]
        offset = transform[:3, 3]
        method = "nifti_affine_world_resampling"
    elif strategy == "foreground_extent":
        matrix, offset = _foreground_extent_transform(
            fixed.data,
            moving.data,
            fixed_mask,
            moving_mask,
            axis_reflections,
        )
        method = "foreground_extent_reflection_affine_resampling"
        matrix, offset, refined = _refine_translation_by_nmi(
            fixed.data,
            moving.data,
            fixed_mask,
            moving_mask,
            matrix,
            offset,
        )
        if refined:
            method += "+nmi_translation_refinement"
    else:
        raise ValueError(f"Unknown affine registration strategy: {strategy}")
    registered = affine_transform(
        moving.data,
        matrix=matrix,
        offset=offset,
        output_shape=fixed.data.shape,
        order=order,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    volume = Volume(name, registered, fixed.spacing_m, fixed.affine, moving.source_path)
    nmi = normalized_mutual_information(fixed.data, registered)
    edge_overlap = binary_edge_overlap(fixed.data, registered)
    return AffineRegistrationResult(
        fixed,
        volume,
        True,
        method,
        nmi,
        edge_overlap,
        (
            f"Affine CT-to-MRI registration executed with {method}; "
            "visual overlay QC is required before clinical use"
        ),
    )


def _foreground_extent_transform(
    fixed_data: np.ndarray,
    moving_data: np.ndarray,
    fixed_mask: np.ndarray | None,
    moving_mask: np.ndarray | None,
    axis_reflections: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    fixed_center, fixed_extent = _mask_center_extent(fixed_data.shape, fixed_mask)
    moving_center, moving_extent = _mask_center_extent(moving_data.shape, moving_mask)
    scale = moving_extent / fixed_extent
    if fixed_mask is None or moving_mask is None or not np.any(fixed_mask) or not np.any(moving_mask):
        matrix = np.diag(scale)
        offset = moving_center - matrix @ fixed_center
        return matrix, offset

    fixed_eval = normalize_unit(fixed_data, fixed_mask)
    best_score = -np.inf
    best_matrix = np.diag(scale)
    best_offset = moving_center - best_matrix @ fixed_center
    for sx in axis_reflections[0]:
        for sy in axis_reflections[1]:
            for sz in axis_reflections[2]:
                matrix = np.diag([sx * scale[0], sy * scale[1], sz * scale[2]])
                offset = moving_center - matrix @ fixed_center
                moved = affine_transform(
                    moving_mask.astype(np.float32),
                    matrix=matrix,
                    offset=offset,
                    output_shape=fixed_data.shape,
                    order=0,
                    mode="constant",
                    cval=0.0,
                ) > 0.5
                moved_data = affine_transform(
                    moving_data,
                    matrix=matrix,
                    offset=offset,
                    output_shape=fixed_data.shape,
                    order=1,
                    mode="constant",
                    cval=0.0,
                ).astype(np.float32)
                moved_eval = normalize_unit(moved_data, fixed_mask)
                score = multimodal_affine_score(fixed_eval, moved_eval, fixed_mask, fixed_mask, moved)
                if score > best_score:
                    best_score = score
                    best_matrix = matrix
                    best_offset = offset
    return best_matrix, best_offset


def _refine_translation_by_nmi(
    fixed_data: np.ndarray,
    moving_data: np.ndarray,
    fixed_mask: np.ndarray | None,
    moving_mask: np.ndarray | None,
    matrix: np.ndarray,
    offset: np.ndarray,
    steps: tuple[int, ...] = (4, 2, 1),
) -> tuple[np.ndarray, np.ndarray, bool]:
    if fixed_mask is None or moving_mask is None or not np.any(fixed_mask) or not np.any(moving_mask):
        return matrix, offset, False
    fixed_eval = normalize_unit(fixed_data, fixed_mask)
    best_offset = np.asarray(offset, dtype=np.float64)
    best_score = _translation_score(fixed_eval, fixed_mask, moving_data, moving_mask, matrix, best_offset)
    refined = False
    for step in steps:
        center = best_offset.copy()
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                for dz in (-step, 0, step):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    candidate = center + np.asarray([dx, dy, dz], dtype=np.float64)
                    score = _translation_score(fixed_eval, fixed_mask, moving_data, moving_mask, matrix, candidate)
                    if score > best_score + 1.0e-9:
                        best_score = score
                        best_offset = candidate
                        refined = True
    return matrix, best_offset, refined


def _translation_score(
    fixed_eval: np.ndarray,
    fixed_mask: np.ndarray,
    moving_data: np.ndarray,
    moving_mask: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
) -> float:
    moved = affine_transform(
        moving_data,
        matrix=matrix,
        offset=offset,
        output_shape=fixed_eval.shape,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    moved_mask = affine_transform(
        moving_mask.astype(np.float32),
        matrix=matrix,
        offset=offset,
        output_shape=fixed_eval.shape,
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5
    metric_mask = fixed_mask & moved_mask
    if np.count_nonzero(metric_mask) < max(8, int(0.25 * np.count_nonzero(fixed_mask))):
        return -np.inf
    moved_eval = normalize_unit(moved, metric_mask)
    nmi = normalized_mutual_information(fixed_eval, moved_eval, mask=metric_mask)
    mse = normalized_mean_squared_error(fixed_eval, moved_eval, metric_mask)
    edge = binary_edge_overlap(fixed_eval * metric_mask, moved_eval * metric_mask)
    dice = dice_overlap(fixed_mask, moved_mask)
    return float(nmi + 0.50 * dice + 0.25 * edge - 0.10 * mse)

def _mask_center_extent(
    shape: tuple[int, int, int],
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if mask is not None and np.any(mask):
        coords = np.argwhere(mask)
        lo = coords.min(axis=0).astype(np.float64)
        hi = coords.max(axis=0).astype(np.float64)
    else:
        lo = np.zeros(3, dtype=np.float64)
        hi = np.asarray(shape, dtype=np.float64) - 1.0
    extent = np.maximum(hi - lo, 1.0)
    center = (lo + hi) * 0.5
    return center, extent


def register_triplet_with_ritk(
    triplet: BrainTriplet,
    max_iterations: int = 12,
) -> RegistrationResult:
    fixed = normalize_unit(triplet.ct_hu.data, triplet.brain_mask | triplet.skull_mask)
    fixed_anatomy_mask = triplet.brain_mask | triplet.skull_mask
    same_patient_rire = (
        triplet.ct_hu.source_path.name == "patient_109_ct.nii.gz"
        and triplet.t1.source_path.name == "patient_109_mr_t1.nii.gz"
    )
    if same_patient_rire:
        t1_affine = affine_register_moving_to_fixed(
            triplet.t1,
            triplet.ct_hu,
            "Affine CT-space RIRE MR-T1",
            strategy="world",
        ).moving_registered
        registration_method = "same_patient_nifti_affine_world_resampling"
    else:
        t1_affine = affine_register_moving_to_fixed(
            triplet.t1,
            triplet.ct_hu,
            "Affine CT-space T1 MRI",
            strategy="foreground_extent",
            fixed_mask=fixed_anatomy_mask,
            moving_mask=foreground_mask(triplet.t1.data),
        ).moving_registered
        registration_method = "foreground_extent_reflection_affine_resampling"
    subject_mri_mask = triplet.brain_mask & foreground_mask(t1_affine.data)
    if not np.any(subject_mri_mask):
        subject_mri_mask = triplet.brain_mask
    atlas_affine = affine_register_moving_to_fixed(
        triplet.atlas_t1,
        t1_affine,
        "Affine subject-MRI-space MNI152 atlas",
        strategy="foreground_extent",
        fixed_mask=subject_mri_mask,
        moving_mask=triplet.atlas_mask.data > 0.5,
        axis_reflections=((1.0,), (1.0,), (1.0,)),
    ).moving_registered
    moving_t1 = normalize_unit(t1_affine.data, triplet.brain_mask)
    moving_atlas = normalize_unit(atlas_affine.data, subject_mri_mask)
    before_t1 = normalized_cross_correlation(fixed, moving_t1, triplet.brain_mask)
    before_atlas = normalized_cross_correlation(moving_t1, moving_atlas, subject_mri_mask)
    before_t1_nmi = normalized_mutual_information(fixed, moving_t1, mask=triplet.brain_mask)
    before_atlas_nmi = normalized_mutual_information(moving_t1, moving_atlas, mask=subject_mri_mask)
    before_t1_mse = normalized_mean_squared_error(fixed, moving_t1, triplet.brain_mask)
    before_atlas_mse = normalized_mean_squared_error(moving_t1, moving_atlas, subject_mri_mask)

    try:
        import ritk  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return RegistrationResult(
            triplet.ct_hu,
            t1_affine,
            atlas_affine,
            False,
            registration_method,
            before_t1,
            before_t1,
            before_t1_nmi,
            before_t1_nmi,
            before_t1_mse,
            before_t1_mse,
            before_atlas,
            before_atlas,
            before_atlas_nmi,
            before_atlas_nmi,
            before_atlas_mse,
            before_atlas_mse,
            f"ritk Python binding unavailable: {exc}",
        )

    if not hasattr(ritk, "registration") or not hasattr(ritk, "Image"):
        return RegistrationResult(
            triplet.ct_hu,
            t1_affine,
            atlas_affine,
            False,
            registration_method,
            before_t1,
            before_t1,
            before_t1_nmi,
            before_t1_nmi,
            before_t1_mse,
            before_t1_mse,
            before_atlas,
            before_atlas,
            before_atlas_nmi,
            before_atlas_nmi,
            before_atlas_mse,
            before_atlas_mse,
            "ritk import resolved to a namespace package without compiled bindings",
        )

    ritk.io.read_image(str(triplet.ct_hu.source_path))
    ritk.io.read_image(str(triplet.t1.source_path))
    ritk.io.read_image(str(triplet.atlas_t1.source_path))
    fixed_img, fixed = _ritk_image_from_volume(
        ritk,
        triplet.ct_hu,
        triplet.brain_mask | triplet.skull_mask,
    )
    t1_img, moving_t1 = _ritk_image_from_volume(ritk, t1_affine, triplet.brain_mask)
    atlas_img, moving_atlas = _ritk_image_from_volume(ritk, atlas_affine, subject_mri_mask)
    before_t1 = normalized_cross_correlation(fixed, moving_t1, triplet.brain_mask)
    before_atlas = normalized_cross_correlation(moving_t1, moving_atlas, subject_mri_mask)
    before_t1_nmi = normalized_mutual_information(fixed, moving_t1, mask=triplet.brain_mask)
    before_atlas_nmi = normalized_mutual_information(moving_t1, moving_atlas, mask=subject_mri_mask)
    before_t1_mse = normalized_mean_squared_error(fixed, moving_t1, triplet.brain_mask)
    before_atlas_mse = normalized_mean_squared_error(moving_t1, moving_atlas, subject_mri_mask)
    iterations = [max(max_iterations // 2, 1), max(max_iterations // 3, 1)]
    _, t1_registered = ritk.registration.multires_syn_register(
        fixed_img,
        t1_img,
        num_levels=2,
        iterations=iterations,
        sigma_smooth=1.5,
        cc_radius=2,
        inverse_consistency=True,
        gradient_step=0.20,
    )
    _, atlas_registered = ritk.registration.multires_syn_register(
        t1_img,
        atlas_img,
        num_levels=2,
        iterations=iterations,
        sigma_smooth=1.5,
        cc_radius=2,
        inverse_consistency=True,
        gradient_step=0.20,
    )
    t1_arr = _ritk_zyx_to_xyz(np.asarray(t1_registered.to_numpy(), dtype=np.float32))
    atlas_arr = _ritk_zyx_to_xyz(np.asarray(atlas_registered.to_numpy(), dtype=np.float32))
    after_t1 = normalized_cross_correlation(fixed, t1_arr, triplet.brain_mask)
    after_atlas = normalized_cross_correlation(moving_t1, atlas_arr, subject_mri_mask)
    after_t1_nmi = normalized_mutual_information(fixed, t1_arr, mask=triplet.brain_mask)
    after_atlas_nmi = normalized_mutual_information(moving_t1, atlas_arr, mask=subject_mri_mask)
    after_t1_mse = normalized_mean_squared_error(fixed, t1_arr, triplet.brain_mask)
    after_atlas_mse = normalized_mean_squared_error(moving_t1, atlas_arr, subject_mri_mask)
    t1_accepted = after_t1_nmi >= before_t1_nmi and registration_quality(after_t1, after_t1_nmi, after_t1_mse) >= registration_quality(
        before_t1, before_t1_nmi, before_t1_mse
    )
    atlas_accepted = after_atlas_nmi >= before_atlas_nmi and registration_quality(
        after_atlas, after_atlas_nmi, after_atlas_mse
    ) >= registration_quality(
        before_atlas, before_atlas_nmi, before_atlas_mse
    )
    if not t1_accepted:
        t1_arr = t1_affine.data
        after_t1, after_t1_nmi, after_t1_mse = before_t1, before_t1_nmi, before_t1_mse
    if not atlas_accepted:
        atlas_arr = atlas_affine.data
        after_atlas, after_atlas_nmi, after_atlas_mse = before_atlas, before_atlas_nmi, before_atlas_mse
    guard = f"metric_guarded_ritk_syn(t1={'accepted' if t1_accepted else 'rejected'},atlas={'accepted' if atlas_accepted else 'rejected'})"
    return RegistrationResult(
        triplet.ct_hu,
        Volume("RITK-registered T1 MRI", t1_arr, triplet.ct_hu.spacing_m, triplet.ct_hu.affine, triplet.t1.source_path),
        Volume(
            "RITK-registered MNI152 atlas",
            atlas_arr,
            triplet.ct_hu.spacing_m,
            triplet.ct_hu.affine,
            triplet.atlas_t1.source_path,
        ),
        True,
        f"{registration_method} + subject_mri_atlas_affine + {guard}",
        before_t1,
        after_t1,
        before_t1_nmi,
        after_t1_nmi,
        before_t1_mse,
        after_t1_mse,
        before_atlas,
        after_atlas,
        before_atlas_nmi,
        after_atlas_nmi,
        before_atlas_mse,
        after_atlas_mse,
        "RITK NIfTI read and metric-guarded registration executed",
    )


def _ritk_image_from_volume(
    ritk: object,
    volume: Volume,
    mask: np.ndarray,
) -> tuple[object, np.ndarray]:
    data = normalize_unit(volume.data, mask)
    spacing_mm = [
        float(volume.spacing_m[2] * 1.0e3),
        float(volume.spacing_m[1] * 1.0e3),
        float(volume.spacing_m[0] * 1.0e3),
    ]
    image = ritk.Image(_xyz_to_ritk_zyx(data), spacing=spacing_mm)
    return image, data


def _xyz_to_ritk_zyx(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(data, (2, 1, 0)), dtype=np.float32)


def _ritk_zyx_to_xyz(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(data, (2, 1, 0)), dtype=np.float32)
