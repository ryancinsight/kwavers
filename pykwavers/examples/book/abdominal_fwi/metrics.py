"""Value-semantic diagnostics for abdominal FWI outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def metric_block(target: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    """Return reconstruction similarity metrics over active voxels."""

    target = np.asarray(target, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    denom = float(np.linalg.norm(target))
    nrmse = float(np.linalg.norm(reconstruction - target) / max(denom, 1.0e-30))
    if float(np.std(target)) == 0.0 or float(np.std(reconstruction)) == 0.0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(target, reconstruction)[0, 1])
    return {
        "pearson_correlation": pearson,
        "nrmse": nrmse,
        "target_dynamic_range": float(np.ptp(target)),
        "reconstruction_dynamic_range": float(np.ptp(reconstruction)),
    }


def equal_volume_dice(values: np.ndarray, truth: np.ndarray, *, negative: bool) -> float:
    """Dice score after selecting the same voxel count as the truth mask."""

    detected = equal_volume_detection(values, truth, negative=negative)
    truth = np.asarray(truth, dtype=bool)
    count = int(np.count_nonzero(truth))
    if count == 0:
        return 0.0
    overlap = int(np.count_nonzero(detected & truth))
    return float(2.0 * overlap / (count + int(np.count_nonzero(detected))))


def equal_volume_detection(
    values: np.ndarray,
    truth: np.ndarray,
    *,
    negative: bool,
) -> np.ndarray:
    """Return detections with cardinality equal to the truth mask."""

    truth = np.asarray(truth, dtype=bool)
    count = int(np.count_nonzero(truth))
    detected = np.zeros(values.shape, dtype=bool)
    if count == 0:
        return detected
    order = np.argsort(values)
    if not negative:
        order = order[::-1]
    detected[order[:count]] = True
    return detected


def contrast_to_noise(values: np.ndarray, mask: np.ndarray) -> float:
    """Signed lesion contrast divided by background standard deviation."""

    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask) or not np.any(~mask):
        return 0.0
    lesion = values[mask]
    background = values[~mask]
    return float(
        abs(float(np.mean(lesion)) - float(np.mean(background)))
        / max(float(np.std(background)), 1.0e-12)
    )


def as_metric_dict(result: Any) -> dict[str, object]:
    """Serialize scalar metrics from a case result."""

    fov_m = [
        float(axis * result.prepared.spacing_m) for axis in result.prepared.ct_hu.shape
    ]
    return {
        "case": result.prepared.name,
        "title": result.prepared.title,
        "source_index": result.prepared.source_index,
        "grid_shape": list(result.prepared.ct_hu.shape),
        "spacing_m": result.prepared.spacing_m,
        "field_of_view_m": fov_m,
        "target_voxels": int(np.count_nonzero(result.prepared.target_mask)),
        "imaging_voxels": int(np.count_nonzero(result.prepared.imaging_mask)),
        "lesion_voxels": int(np.count_nonzero(result.lesion_mask)),
        "simulation": result.simulation.metrics,
        "targeting": result.targeting.metrics,
        "lesioning": result.lesioning.metrics,
        "subharmonic": result.subharmonic.metrics,
        "nonlinear": result.nonlinear.metrics,
        "model_limits": (
            "Synthetic 2-D Born receiver inversion on CT-derived abdominal slices "
            "with bounded 2-D Westervelt and Rayleigh-Plesset source maps; not "
            "measured histotripsy hardware data, not 3-D adjoint nonlinear FWI, "
            "and not clinical targeting validation."
        ),
    }
