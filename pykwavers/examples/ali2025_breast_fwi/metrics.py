"""Metric contracts for Ali 2025 reduced-grid replication."""

from __future__ import annotations

import numpy as np

from .volume import paired_arrays


ALI_2025_TABLE1_3D_FWI = {
    1: {"rmse_m_s": 15.5, "pearson_correlation": 0.8848},
    2: {"rmse_m_s": 10.1, "pearson_correlation": 0.8981},
    3: {"rmse_m_s": 8.4, "pearson_correlation": 0.8967},
}


def rmse_m_s(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref, est = paired_arrays(reference, estimate)
    return float(np.sqrt(np.mean(np.square(est - ref))))


def normalized_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref, _ = paired_arrays(reference, estimate)
    scale = float(np.sqrt(np.mean(np.square(ref))))
    if scale <= 0.0:
        raise ValueError("reference RMS is zero")
    return rmse_m_s(reference, estimate) / scale


def pearson_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref, est = paired_arrays(reference, estimate)
    ref_centered = ref - np.mean(ref)
    est_centered = est - np.mean(est)
    denom = float(np.linalg.norm(ref_centered) * np.linalg.norm(est_centered))
    if denom <= 0.0:
        raise ValueError("Pearson correlation requires nonconstant inputs")
    return float(np.vdot(ref_centered, est_centered).real / denom)


def reconstruction_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref, est = paired_arrays(reference, estimate)
    rmse = float(np.sqrt(np.mean(np.square(est - ref))))
    reference_rms = float(np.sqrt(np.mean(np.square(ref))))
    ref_centered = ref - np.mean(ref)
    est_centered = est - np.mean(est)
    correlation_denominator = float(np.linalg.norm(ref_centered) * np.linalg.norm(est_centered))
    if reference_rms <= 0.0:
        raise ValueError("reference RMS is zero")
    if correlation_denominator <= 0.0:
        raise ValueError("Pearson correlation requires nonconstant inputs")
    return {
        "rmse_m_s": rmse,
        "normalized_rmse": rmse / reference_rms,
        "pearson_correlation": float(np.vdot(ref_centered, est_centered).real / correlation_denominator),
        "reference_min_m_s": float(np.min(ref)),
        "reference_max_m_s": float(np.max(ref)),
        "estimate_min_m_s": float(np.min(est)),
        "estimate_max_m_s": float(np.max(est)),
    }


def table1_parity(
    metrics: dict[str, float],
    phantom_index: int,
    rmse_multiplier: float,
    pcc_fraction: float,
) -> dict[str, float | bool]:
    if phantom_index not in ALI_2025_TABLE1_3D_FWI:
        raise ValueError(f"phantom_index must be one of 1, 2, 3, got {phantom_index}")
    if not np.isfinite(rmse_multiplier) or rmse_multiplier <= 0.0:
        raise ValueError(f"rmse_multiplier must be positive and finite, got {rmse_multiplier}")
    if not np.isfinite(pcc_fraction) or pcc_fraction <= 0.0:
        raise ValueError(f"pcc_fraction must be positive and finite, got {pcc_fraction}")

    reference = ALI_2025_TABLE1_3D_FWI[phantom_index]
    rmse_threshold = reference["rmse_m_s"] * rmse_multiplier
    pcc_threshold = reference["pearson_correlation"] * pcc_fraction
    rmse = float(metrics["rmse_m_s"])
    pcc = float(metrics["pearson_correlation"])
    rmse_pass = rmse <= rmse_threshold
    pcc_pass = pcc >= pcc_threshold
    return {
        "phantom_index": float(phantom_index),
        "table1_3d_rmse_m_s": reference["rmse_m_s"],
        "table1_3d_pearson_correlation": reference["pearson_correlation"],
        "rmse_threshold_m_s": rmse_threshold,
        "pcc_threshold": pcc_threshold,
        "rmse_pass": rmse_pass,
        "pcc_pass": pcc_pass,
        "passes": rmse_pass and pcc_pass,
    }
