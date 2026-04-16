"""
Shared utilities for pykwavers vs k-wave-python example parity scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).parents[2]
KWAVE_PYTHON_ROOT = ROOT / "external" / "k-wave-python"
PYKWAVERS_PYTHON_ROOT = ROOT / "pykwavers" / "python"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def bootstrap_example_paths() -> Path:
    """Ensure local k-wave-python and pykwavers Python packages are importable."""
    for path in (KWAVE_PYTHON_ROOT, PYKWAVERS_PYTHON_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT


def normalize_sensor_matrix(data: np.ndarray, expected_sensors: int | None = None) -> np.ndarray:
    """Normalize sensor data to shape `(n_sensors, n_time_samples)`."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if expected_sensors is not None:
        if arr.shape[0] == expected_sensors:
            return arr
        if arr.shape[1] == expected_sensors:
            return arr.T
        raise ValueError(
            f"sensor matrix shape {arr.shape} does not match expected sensor count {expected_sensors}"
        )
    if arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def compute_trace_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compute parity metrics for two 1-D traces."""
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    n = min(ref.size, cand.size)
    ref = ref[:n]
    cand = cand[:n]

    ref_std = float(np.std(ref))
    cand_std = float(np.std(cand))
    if ref_std > 1e-30 and cand_std > 1e-30:
        corr = float(np.corrcoef(ref, cand)[0, 1])
    else:
        corr = 0.0

    ref_rms = float(np.sqrt(np.mean(ref**2)))
    cand_rms = float(np.sqrt(np.mean(cand**2)))
    rms_ratio = cand_rms / (ref_rms + 1e-30)

    diff = cand - ref
    max_abs_diff = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ref_peak = float(np.max(np.abs(ref)))
    cand_peak = float(np.max(np.abs(cand)))
    peak_ratio = cand_peak / (ref_peak + 1e-30)

    return {
        "pearson_r": corr,
        "rms_ratio": rms_ratio,
        "rmse": rmse,
        "max_abs_diff": max_abs_diff,
        "reference_peak": ref_peak,
        "candidate_peak": cand_peak,
        "peak_ratio": peak_ratio,
    }


def compute_image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compute Pearson-r, RMS ratio, and PSNR for two images/2-D arrays."""
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    n = min(ref.size, cand.size)
    ref = ref[:n]
    cand = cand[:n]

    corr = float(np.corrcoef(ref, cand)[0, 1])
    ref_rms = float(np.sqrt(np.mean(ref**2)))
    cand_rms = float(np.sqrt(np.mean(cand**2)))
    rms_ratio = cand_rms / (ref_rms + 1e-30)
    mse = float(np.mean((ref - cand) ** 2))
    peak = float(max(np.max(np.abs(ref)), np.max(np.abs(cand))))
    psnr_db = 20.0 * np.log10(peak / (np.sqrt(mse) + 1e-30))

    return {
        "pearson_r": corr,
        "rms_ratio": rms_ratio,
        "psnr_db": psnr_db,
    }


def save_text_report(path: Path, header: str, lines: Iterable[str]) -> None:
    """Write a plain-text report with a fixed header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header.rstrip() + "\n")
        for line in lines:
            handle.write(str(line).rstrip() + "\n")
