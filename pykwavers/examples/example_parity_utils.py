"""
Shared utilities for pykwavers vs k-wave-python example parity scripts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).parents[2]
KWAVE_PYTHON_ROOT = ROOT / "external" / "k-wave-python"
PYKWAVERS_PYTHON_ROOT = ROOT / "pykwavers" / "python"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


class RunningTimingStats:
    """Accumulate per-line timing tuples without materializing per-line dicts."""

    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = labels
        self.count = 0
        size = len(self.labels)
        self.sums_ns = np.zeros(size, dtype=np.float64)
        self.mins_ns = np.full(size, np.inf, dtype=np.float64)
        self.maxs_ns = np.zeros(size, dtype=np.float64)

    def update(self, values: Sequence[float]) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != len(self.labels):
            raise ValueError(
                f"Expected {len(self.labels)} timing values, got shape {arr.shape}"
            )
        self.count += 1
        self.sums_ns += arr
        np.minimum(self.mins_ns, arr, out=self.mins_ns)
        np.maximum(self.maxs_ns, arr, out=self.maxs_ns)

    def summary_ms(self) -> dict[str, dict[str, float]]:
        if self.count == 0:
            raise ValueError("No timing samples recorded")
        mean_ms = self.sums_ns / self.count / 1e6
        min_ms = self.mins_ns / 1e6
        max_ms = self.maxs_ns / 1e6
        summary: dict[str, dict[str, float]] = {}
        for idx, label in enumerate(self.labels):
            summary[label] = {
                "mean": float(mean_ms[idx]),
                "min": float(min_ms[idx]),
                "max": float(max_ms[idx]),
            }
        return summary

    def format_lines(self, indent: str = "  ") -> list[str]:
        summary = self.summary_ms()
        width = max(len(label) for label in self.labels)
        lines: list[str] = []
        for label in self.labels:
            stats = summary[label]
            lines.append(
                f"{indent}{label:<{width}}  "
                f"mean={stats['mean']:.3f} max={stats['max']:.3f} min={stats['min']:.3f}"
            )
        return lines


def bootstrap_example_paths() -> Path:
    """Ensure local k-wave-python and pykwavers Python packages are importable."""
    extension_candidates = (
        ROOT / "target" / "maturin" / "pykwavers.dll",
        ROOT / "target" / "release" / "pykwavers.dll",
        ROOT / "target" / "debug" / "deps" / "pykwavers.dll",
        ROOT / "target" / "debug" / "pykwavers.dll",
        PYKWAVERS_PYTHON_ROOT / "pykwavers" / "_pykwavers.pyd",
    )
    for extension_path in extension_candidates:
        if extension_path.exists():
            os.environ["PYKWAVERS_EXTENSION_PATH"] = str(extension_path)
            break
    for path in (KWAVE_PYTHON_ROOT, PYKWAVERS_PYTHON_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT


def _normalize_pml_size(pml_size: tuple[int, ...], is_2d: bool) -> tuple[int, int, int]:
    if len(pml_size) == 2:
        if not is_2d:
            raise ValueError(
                f"Expected a 3-D PML tuple for a 3-D volume, got 2 values: {pml_size}"
            )
        px, py = pml_size
        return (int(px), int(py), 0)
    if len(pml_size) == 3:
        px, py, pz = pml_size
        return (int(px), int(py), 0 if is_2d else int(pz))
    raise ValueError(f"Expected a 2-D or 3-D PML tuple, got {pml_size}")


def expand_pml_outside_shape(shape: tuple[int, ...], pml_size: tuple[int, ...]) -> tuple[int, int, int]:
    """Return the total grid shape after embedding the physical domain inside outer PML layers."""
    if len(shape) == 2:
        nx, ny = shape
        nz = 1
        px, py, pz = _normalize_pml_size(pml_size, is_2d=True)
    elif len(shape) == 3:
        nx, ny, nz = shape
        px, py, pz = _normalize_pml_size(pml_size, is_2d=False)
    else:
        raise ValueError(f"Expected a 2-D or 3-D shape, got {shape}")
    return (nx + 2 * px, ny + 2 * py, nz + 2 * pz)


def pad_volume_for_pml_outside(volume: np.ndarray, pml_size: tuple[int, ...]) -> np.ndarray:
    """Zero-pad a 2-D or 3-D volume so the active region sits inside outer PML layers."""
    arr = np.asarray(volume)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D volume, got shape {arr.shape}")

    px, py, pz = _normalize_pml_size(pml_size, is_2d=arr.shape[2] == 1)
    pad_width = ((px, px), (py, py), (pz, pz))
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def clip_volume_to_physical_interior(volume: np.ndarray, pml_size: tuple[int, ...]) -> np.ndarray:
    """Zero the outer PML halo of a padded 3-D volume while preserving shape."""
    arr = np.asarray(volume).copy()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D volume, got shape {arr.shape}")

    px, py, pz = _normalize_pml_size(pml_size, is_2d=arr.shape[2] == 1)
    if px:
        arr[:px, :, :] = 0
        arr[-px:, :, :] = 0
    if py:
        arr[:, :py, :] = 0
        arr[:, -py:, :] = 0
    if pz:
        arr[:, :, :pz] = 0
        arr[:, :, -pz:] = 0
    return arr


def advance_lateral_window_inplace(
    current_volume: np.ndarray,
    source_volume: np.ndarray,
    previous_start: int,
    current_start: int,
) -> None:
    """Advance a sliding y-axis window in place.

    The current volume is the active window stored inside the padded full-grid
    buffer. Advancing the window only copies the new columns that enter on the
    right-hand side and shifts the overlap in place.
    """
    if current_volume.ndim != 3 or source_volume.ndim != 3:
        raise ValueError(
            "advance_lateral_window_inplace expects 3-D current and source volumes"
        )
    width = current_volume.shape[1]
    delta = int(current_start) - int(previous_start)
    if delta < 0:
        raise ValueError(
            f"current_start must be >= previous_start, got {current_start} < {previous_start}"
        )
    if delta == 0:
        return
    if delta >= width:
        np.copyto(current_volume, source_volume[:, current_start : current_start + width, :])
        return

    np.copyto(current_volume[:, : width - delta, :], current_volume[:, delta:width, :])
    np.copyto(
        current_volume[:, width - delta :, :],
        source_volume[:, current_start + width - delta : current_start + width, :],
    )


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


def summarize_sensor_matrix_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    expected_sensors: int | None = None,
) -> dict[str, float]:
    """Summarize parity metrics for a sensor matrix pair.

    The matrices are normalized to ``(n_sensors, n_time_samples)`` and then
    compared row-wise. This matches the k-Wave ordering contract used by the
    current sensor recorder implementation.
    """
    ref = normalize_sensor_matrix(reference, expected_sensors=expected_sensors)
    cand = normalize_sensor_matrix(candidate, expected_sensors=ref.shape[0])
    if ref.shape != cand.shape:
        raise ValueError(f"sensor matrix shape mismatch: {ref.shape} != {cand.shape}")

    row_metrics = [compute_trace_metrics(ref[i], cand[i]) for i in range(ref.shape[0])]
    pearson_r = np.asarray([m["pearson_r"] for m in row_metrics], dtype=float)
    rms_ratio = np.asarray([m["rms_ratio"] for m in row_metrics], dtype=float)
    rmse = np.asarray([m["rmse"] for m in row_metrics], dtype=float)
    max_abs_diff = np.asarray([m["max_abs_diff"] for m in row_metrics], dtype=float)
    peak_ratio = np.asarray([m["peak_ratio"] for m in row_metrics], dtype=float)

    return {
        "n_sensors": float(ref.shape[0]),
        "n_time_samples": float(ref.shape[1]),
        "pearson_r_mean": float(np.mean(pearson_r)),
        "pearson_r_median": float(np.median(pearson_r)),
        "rms_ratio_mean": float(np.mean(rms_ratio)),
        "rms_ratio_median": float(np.median(rms_ratio)),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_median": float(np.median(rmse)),
        "max_abs_diff_max": float(np.max(max_abs_diff)),
        "peak_ratio_mean": float(np.mean(peak_ratio)),
        "peak_ratio_median": float(np.median(peak_ratio)),
    }


def compute_image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compute parity metrics for two images/2-D arrays."""
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    n = min(ref.size, cand.size)
    ref = ref[:n]
    cand = cand[:n]

    corr = float(np.corrcoef(ref, cand)[0, 1])
    ref_rms = float(np.sqrt(np.mean(ref**2)))
    cand_rms = float(np.sqrt(np.mean(cand**2)))
    rms_ratio = cand_rms / (ref_rms + 1e-30)
    diff = cand - ref
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    ref_peak = float(np.max(np.abs(ref)))
    cand_peak = float(np.max(np.abs(cand)))
    peak_ratio = cand_peak / (ref_peak + 1e-30)
    peak = float(max(ref_peak, cand_peak))
    psnr_db = float(20.0 * np.log10(peak / (rmse + 1e-30)))

    return {
        "pearson_r": corr,
        "rms_ratio": rms_ratio,
        "rmse": rmse,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "reference_peak": ref_peak,
        "candidate_peak": cand_peak,
        "peak_ratio": peak_ratio,
        "psnr_db": psnr_db,
    }


def save_text_report(path: Path, header: str, lines: Iterable[str]) -> None:
    """Write a plain-text report with a fixed header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header.rstrip() + "\n")
        for line in lines:
            handle.write(str(line).rstrip() + "\n")
