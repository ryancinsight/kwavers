"""Artifact rendering for the `at_array_as_source` parity example."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _reshape_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(field, dtype=np.float64).reshape(shape, order="C")


def save_comparison_figure(result: dict[str, object], figure_path: Path) -> None:
    """Render the source geometry, source distribution, and derived fields."""
    kw = result["kwave"]  # type: ignore[assignment]
    py = result["pykwavers"]  # type: ignore[assignment]
    layout = result["layout"]  # type: ignore[assignment]

    source_binary_mask = np.asarray(layout["source_binary_mask"], dtype=bool)  # type: ignore[index]
    source_weighted_mask = np.asarray(layout["source_weighted_mask"], dtype=np.float64)  # type: ignore[index]
    source_signal_kw = np.asarray(layout["source_signal_kw"], dtype=np.float64)  # type: ignore[index]
    source_signal_py = np.asarray(layout["source_signal_py"], dtype=np.float64)  # type: ignore[index]
    field_shape = source_binary_mask.shape
    p_max_kw = _reshape_field(np.asarray(kw["p_max"], dtype=np.float64), field_shape)  # type: ignore[index]
    p_max_py = _reshape_field(np.asarray(py["p_max"], dtype=np.float64), field_shape)  # type: ignore[index]
    p_rms_kw = _reshape_field(np.asarray(kw["p_rms"], dtype=np.float64), field_shape)  # type: ignore[index]
    p_rms_py = _reshape_field(np.asarray(py["p_rms"], dtype=np.float64), field_shape)  # type: ignore[index]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    im = ax.imshow(
        source_weighted_mask.T,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.contour(source_binary_mask.T.astype(np.float64), levels=[0.5], colors="white", linewidths=0.8)
    ax.set_title("Source weighted mask")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(
        source_signal_kw - source_signal_py,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_title("Distributed source signal difference")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Active source row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(
        p_rms_kw - p_rms_py,
        origin="lower",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_title("p_rms difference")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    im = ax.imshow(
        p_max_kw - p_max_py,
        origin="lower",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_title("p_max difference")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_report_lines(result: dict[str, object]) -> list[str]:
    metrics = result["metrics"]  # type: ignore[assignment]
    kw = result["kwave"]  # type: ignore[assignment]
    layout = result["layout"]  # type: ignore[assignment]
    source_shape = np.asarray(layout["source_binary_mask"], dtype=bool).shape  # type: ignore[index]
    return [
        "example: at_array_as_source",
        f"grid: {source_shape[0]}x{source_shape[1]}",
        f"dt_s: {float(kw['dt']):.9e}",  # type: ignore[index]
        f"kwave_runtime_s: {float(kw['runtime_s']):.3f}",  # type: ignore[index]
        f"pykwavers_runtime_s: {float(result['pykwavers']['runtime_s']):.3f}",  # type: ignore[index]
        "",
        "source mask:",
        f"  pearson_r = {metrics['source_mask']['pearson_r']:.6f}",
        f"  rms_ratio = {metrics['source_mask']['rms_ratio']:.6f}",
        f"  rmse      = {metrics['source_mask']['rmse']:.6e}",
        f"  max_abs_diff = {metrics['source_mask']['max_abs_diff']:.6e}",
        f"  peak_ratio   = {metrics['source_mask']['peak_ratio']:.6f}",
        "",
        "source weighted mask:",
        f"  pearson_r = {metrics['source_weighted_mask']['pearson_r']:.6f}",
        f"  rms_ratio = {metrics['source_weighted_mask']['rms_ratio']:.6f}",
        f"  rmse      = {metrics['source_weighted_mask']['rmse']:.6e}",
        f"  max_abs_diff = {metrics['source_weighted_mask']['max_abs_diff']:.6e}",
        f"  peak_ratio   = {metrics['source_weighted_mask']['peak_ratio']:.6f}",
        "",
        "distributed source signal:",
        f"  pearson_r = {metrics['source_signal']['pearson_r']:.6f}",
        f"  rms_ratio = {metrics['source_signal']['rms_ratio']:.6f}",
        f"  rmse      = {metrics['source_signal']['rmse']:.6e}",
        f"  max_abs_diff = {metrics['source_signal']['max_abs_diff']:.6e}",
        f"  peak_ratio   = {metrics['source_signal']['peak_ratio']:.6f}",
        "",
        "p_rms:",
        f"  pearson_r = {metrics['p_rms']['pearson_r']:.6f}",
        f"  rms_ratio = {metrics['p_rms']['rms_ratio']:.6f}",
        f"  rmse      = {metrics['p_rms']['rmse']:.6e}",
        f"  max_abs_diff = {metrics['p_rms']['max_abs_diff']:.6e}",
        f"  peak_ratio   = {metrics['p_rms']['peak_ratio']:.6f}",
        f"  psnr_db   = {metrics['p_rms']['psnr_db']:.6f}",
        "",
        "p_max:",
        f"  pearson_r = {metrics['p_max']['pearson_r']:.6f}",
        f"  rms_ratio = {metrics['p_max']['rms_ratio']:.6f}",
        f"  rmse      = {metrics['p_max']['rmse']:.6e}",
        f"  max_abs_diff = {metrics['p_max']['max_abs_diff']:.6e}",
        f"  peak_ratio   = {metrics['p_max']['peak_ratio']:.6f}",
        f"  psnr_db   = {metrics['p_max']['psnr_db']:.6f}",
    ]
