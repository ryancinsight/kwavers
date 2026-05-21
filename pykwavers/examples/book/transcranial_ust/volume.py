"""Volume slicing and metric helpers for Chapter 27."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

VISIBILITY_FRACTION = 0.35
MIN_OBJECTIVE_REDUCTION = 0.50


def metrics_dict(result: dict) -> dict[str, float | int]:
    metrics = dict(result["metrics"])
    out: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key in {"active_voxels", "measurements", "continuation_stages"}:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def finite_range(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return 0.0
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    return float(np.sum(x0 * y0) / denom) if denom > 0.0 else 0.0


def slice_metrics(slice_result: dict, global_metrics: dict[str, float | int]) -> dict[str, float | int]:
    mask = np.asarray(slice_result["brain_mask"], dtype=bool)
    target = np.asarray(slice_result["target_sound_speed_m_s"], dtype=float)[mask]
    recon = np.asarray(slice_result["reconstruction_sound_speed_m_s"], dtype=float)[mask]
    migration = np.asarray(slice_result["migration_sound_speed_m_s"], dtype=float)[mask]
    enhanced = np.asarray(slice_result["enhanced_reconstruction_sound_speed_m_s"], dtype=float)[mask]
    if target.size == 0:
        return {
            "pearson_correlation": 0.0,
            "normalized_rmse": 0.0,
            "migration_pearson_correlation": 0.0,
            "migration_normalized_rmse": 0.0,
            "migration_dynamic_range_m_s": 0.0,
            "enhanced_dynamic_range_m_s": 0.0,
            "target_dynamic_range_m_s": 0.0,
            "reconstruction_dynamic_range_m_s": 0.0,
            "objective_reduction_fraction": float(global_metrics["objective_reduction_fraction"]),
            "active_voxels": 0,
        }
    norm = max(float(np.sqrt(np.sum(target * target))), 1.0e-12)
    return {
        "pearson_correlation": pearson(target, recon),
        "normalized_rmse": float(np.sqrt(np.sum((target - recon) ** 2)) / norm),
        "migration_pearson_correlation": pearson(target, migration),
        "migration_normalized_rmse": float(np.sqrt(np.sum((target - migration) ** 2)) / norm),
        "migration_dynamic_range_m_s": finite_range(migration),
        "enhanced_dynamic_range_m_s": finite_range(enhanced),
        "target_dynamic_range_m_s": finite_range(target),
        "reconstruction_dynamic_range_m_s": finite_range(recon),
        "objective_reduction_fraction": float(global_metrics["objective_reduction_fraction"]),
        "active_voxels": int(target.size),
    }


def slice_volume_result(
    volume_result: dict,
    volume_index: int,
    global_metrics: dict[str, float | int],
) -> dict:
    sliceable = {
        "ct_hu",
        "target_sound_speed_m_s",
        "initial_sound_speed_m_s",
        "migration_sound_speed_m_s",
        "reconstruction_sound_speed_m_s",
        "enhanced_reconstruction_sound_speed_m_s",
        "brain_mask",
        "skull_mask",
    }
    sliced = {key: value for key, value in volume_result.items() if key not in sliceable}
    for key in sliceable:
        sliced[key] = np.asarray(volume_result[key])[:, :, volume_index]
    spacing_m = float(volume_result["spacing_m"])
    nz = int(np.asarray(volume_result["ct_hu"]).shape[2])
    sliced["source_volume_index"] = int(volume_index)
    sliced["source_slice_index"] = int(volume_index)
    sliced["slice_offset_m"] = (volume_index - 0.5 * (nz - 1)) * spacing_m
    sliced["metrics"] = slice_metrics(sliced, global_metrics)
    return sliced


def visible_reconstruction(metrics: dict[str, float | int]) -> bool:
    target_range = float(metrics["target_dynamic_range_m_s"])
    recon_range = float(metrics["reconstruction_dynamic_range_m_s"])
    reduction = float(metrics["objective_reduction_fraction"])
    return (
        reduction >= MIN_OBJECTIVE_REDUCTION
        and target_range > 0.0
        and recon_range / target_range >= VISIBILITY_FRACTION
    )


def shift_no_wrap(values: np.ndarray, axis: int, offset: int) -> np.ndarray:
    out = np.zeros_like(values)
    if axis == 0 and offset > 0:
        out[offset:, :] = values[:-offset, :]
    elif axis == 0:
        out[:offset, :] = values[-offset:, :]
    elif offset > 0:
        out[:, offset:] = values[:, :-offset]
    else:
        out[:, :offset] = values[:, -offset:]
    return out


def masked_diffusion(image: np.ndarray, mask: np.ndarray, passes: int) -> np.ndarray:
    out = np.where(mask, image, 0.0).astype(float)
    support = mask.astype(float)
    for _ in range(passes):
        acc = out.copy()
        count = support.copy()
        for axis, offset in ((0, 1), (0, -1), (1, 1), (1, -1)):
            shifted_support = shift_no_wrap(support, axis, offset)
            acc += shift_no_wrap(out, axis, offset) * shifted_support
            count += shifted_support
        out = np.where(mask, acc / np.maximum(count, 1.0), 0.0)
    return out


def regularized_fwi_display(reconstruction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    smooth = masked_diffusion(reconstruction, mask, 2)
    detail = np.where(mask, reconstruction - smooth, 0.0)
    values = np.abs(detail[mask])
    if values.size == 0:
        return reconstruction
    limit = max(float(np.percentile(values, 85)), 1.0e-9)
    display = smooth + 0.25 * np.clip(detail, -limit, limit)
    return np.where(mask, display, reconstruction)


def synthetic_data_tensor(result: dict) -> np.ndarray:
    data = np.asarray(result["synthetic_data"], dtype=float)
    element_count = int(result["element_count"])
    offset_count = len(result["receiver_offsets"])
    frequency_count = len(result["frequencies_hz"])
    harmonic_count = int(result.get("harmonic_count", 1))
    expected = element_count * offset_count * frequency_count * harmonic_count
    if data.size != expected:
        raise ValueError(
            "synthetic_data size does not match acquisition dimensions: "
            f"got {data.size}, expected {expected}"
        )
    return data.reshape(element_count, offset_count, frequency_count, harmonic_count)


def extent_mm(result: dict) -> tuple[float, float, float, float]:
    ct = np.asarray(result["ct_hu"])
    spacing_mm = 1.0e3 * float(result["spacing_m"])
    nx, ny = ct.shape
    return (-0.5 * nx * spacing_mm, 0.5 * nx * spacing_mm, -0.5 * ny * spacing_mm, 0.5 * ny * spacing_mm)


def hemispherical_projection_mm(element_count: int, radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(element_count, dtype=float)
    z = radius_mm * (idx + 0.5) / element_count
    radial = np.sqrt(np.maximum(radius_mm * radius_mm - z * z, 0.0))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * idx
    return radial * np.cos(phi), radial * np.sin(phi)


def contour_mask(ax: plt.Axes, mask: np.ndarray, extent: tuple[float, float, float, float], color: str) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors=[color], linewidths=0.8)
