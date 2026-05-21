"""
Chapter 27: Transcranial UST Brain Imaging
===============================================

Executable figure generation for docs/book/transcranial_ust_brain_imaging.md.

The script runs the RITK-backed pykwavers wrapper over the local RIRE patient
109 head CT.  The inversion core lives in kwavers Rust; Python only selects the
chapter parameters, writes figures, and records metrics.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (
    REPO_ROOT / "target" / "release" / "pykwavers.dll",
    REPO_ROOT / "target" / "maturin" / "pykwavers.dll",
    REPO_ROOT / "target" / "debug" / "pykwavers.dll",
):
    if candidate.exists():
        os.environ["PYKWAVERS_EXTENSION_PATH"] = str(candidate)
        break
PY_PACKAGE = REPO_ROOT / "pykwavers" / "python"
if str(PY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PY_PACKAGE))

import pykwavers as kw  # noqa: E402
from transcranial_ust.roi import plot_centroid_roi_stack  # noqa: E402
from transcranial_ust.volume import (  # noqa: E402
    MIN_OBJECTIVE_REDUCTION,
    VISIBILITY_FRACTION,
    contour_mask,
    extent_mm,
    hemispherical_projection_mm,
    metrics_dict,
    regularized_fwi_display,
    slice_volume_result,
    synthetic_data_tensor,
    visible_reconstruction,
)

OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch27"
CT_PATH = REPO_ROOT / "data" / "rire_patient_109" / "patient_109_ct.nii.gz"

FALSE_VALUES = {"0", "false", "no", "off"}


def env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in FALSE_VALUES


GRID_SIZE = int(os.environ.get("KWAVERS_CH27_GRID_SIZE", "56"))
FREQUENCIES_HZ = tuple(
    float(v)
    for v in os.environ.get(
        "KWAVERS_CH27_FREQUENCIES_HZ",
        "200000,350000,500000,650000,800000",
    ).split(",")
)
RECEIVER_OFFSETS = tuple(
    int(v)
    for v in os.environ.get(
        "KWAVERS_CH27_RECEIVER_OFFSETS",
        "512,384,640,256,768,128,448,576",
    ).split(",")
)
ITERATION_SCHEDULE = tuple(
    int(v) for v in os.environ.get("KWAVERS_CH27_ITERATIONS", "12").split(",")
)
STACK_SLICE_OFFSETS = tuple(
    int(v)
    for v in os.environ.get(
        "KWAVERS_CH27_STACK_OFFSETS",
        "-8,-6,-4,-2,0,2,4,6,8,10,12,14,16",
    ).split(",")
)
CENTROID_ROI_HALF_WIDTH_MM = float(os.environ.get("KWAVERS_CH27_CENTROID_ROI_HALF_WIDTH_MM", "35"))
ATTENUATION_MODEL = env_flag("KWAVERS_CH27_ATTENUATION_MODEL", "1")
NONLINEAR_HARMONIC_MODEL = env_flag("KWAVERS_CH27_NONLINEAR_HARMONIC_MODEL", "1")
SOURCE_PRESSURE_MPA = float(os.environ.get("KWAVERS_CH27_SOURCE_PRESSURE_MPA", "0.15"))
NONLINEAR_BETA = float(os.environ.get("KWAVERS_CH27_NONLINEAR_BETA", "4.5"))
EDGE_PRESERVING_WEIGHT = float(os.environ.get("KWAVERS_CH27_EDGE_PRESERVING_WEIGHT", "0.0001"))
EDGE_PRESERVING_EPSILON = float(os.environ.get("KWAVERS_CH27_EDGE_PRESERVING_EPSILON", "0.004"))
EDGE_PRESERVING_STEP = float(os.environ.get("KWAVERS_CH27_EDGE_PRESERVING_STEP", "0.12"))
EDGE_PRESERVING_ITERATIONS = int(os.environ.get("KWAVERS_CH27_EDGE_PRESERVING_ITERATIONS", "1"))

def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch27/{name}.{{pdf,png}}")

def run_fwi_schedule() -> tuple[dict, dict[str, float | int], bool, int]:
    best_result: dict | None = None
    best_metrics: dict[str, float | int] | None = None
    best_score = -np.inf
    for iterations in ITERATION_SCHEDULE:
        print(f"[ch27] Running RITK CT -> 1024-element 3-D bowl-array FWI, iterations={iterations}")
        result = kw.run_transcranial_ust_volume_inversion_from_ritk_ct(
            str(CT_PATH),
            grid_size=GRID_SIZE,
            iterations=iterations,
            frequencies_hz=list(FREQUENCIES_HZ),
            receiver_offsets=list(RECEIVER_OFFSETS),
            edge_preserving_weight=EDGE_PRESERVING_WEIGHT,
            edge_preserving_epsilon=EDGE_PRESERVING_EPSILON,
            edge_preserving_step=EDGE_PRESERVING_STEP,
            edge_preserving_iterations=EDGE_PRESERVING_ITERATIONS,
            attenuation_model=ATTENUATION_MODEL,
            nonlinear_harmonic_model=NONLINEAR_HARMONIC_MODEL,
            source_pressure_mpa=SOURCE_PRESSURE_MPA,
            nonlinear_beta=NONLINEAR_BETA,
        )
        metrics = metrics_dict(result)
        score = float(metrics["objective_reduction_fraction"]) + float(
            metrics["reconstruction_dynamic_range_m_s"]
        ) / max(float(metrics["target_dynamic_range_m_s"]), 1.0e-12)
        if score > best_score:
            best_result = result
            best_metrics = metrics
            best_score = score
        if visible_reconstruction(metrics):
            return result, metrics, True, iterations

    assert best_result is not None and best_metrics is not None
    return best_result, best_metrics, False, int(ITERATION_SCHEDULE[-1])


def run_multislice_stack(
    volume_result: dict,
    global_metrics: dict[str, float | int],
) -> list[tuple[dict, dict[str, float | int]]]:
    center_slice = int(volume_result["source_volume_index"])
    nz = int(np.asarray(volume_result["ct_hu"]).shape[2])
    stack: list[tuple[dict, dict[str, float | int]]] = []
    seen: set[int] = set()
    for offset in STACK_SLICE_OFFSETS:
        slice_index = center_slice + offset
        if slice_index < 0 or slice_index >= nz or slice_index in seen:
            continue
        seen.add(slice_index)
        result = slice_volume_result(volume_result, slice_index, global_metrics)
        if int(result["metrics"]["active_voxels"]) == 0:
            continue
        stack.append((result, result["metrics"]))
    if len(stack) < 3:
        raise RuntimeError(f"multi-slice visualization requires at least 3 valid slices, got {len(stack)}")
    return stack

def plot_ct_and_geometry(result: dict) -> None:
    ct = np.asarray(result["ct_hu"], dtype=float)
    brain_mask = np.asarray(result["brain_mask"], dtype=bool)
    skull_mask = np.asarray(result["skull_mask"], dtype=bool)
    extent = extent_mm(result)
    radius_mm = 1.0e3 * float(result["radius_m"])
    x_elem, y_elem = hemispherical_projection_mm(int(result["element_count"]), radius_mm)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    im = ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-150, vmax=900)
    ax.scatter(x_elem, y_elem, s=2.0, c="#2a6fbb", alpha=0.72)
    contour_mask(ax, skull_mask, extent, "#f28e2b")
    contour_mask(ax, brain_mask, extent, "#59a14f")
    ax.set_title("RITK CT volume slice and 1024-element hemispherical projection")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label="HU")
    savefig("fig01_ct_geometry")
    plt.close(fig)


def plot_acoustic_model(result: dict) -> None:
    target = np.asarray(result["target_sound_speed_m_s"], dtype=float)
    initial = np.asarray(result["initial_sound_speed_m_s"], dtype=float)
    brain = np.asarray(result["brain_mask"], dtype=bool)
    extent = extent_mm(result)
    vmin = np.percentile(target[brain], 2)
    vmax = np.percentile(target[brain], 98)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)
    for ax, image, title in zip(
        axes,
        (target, initial),
        ("CT-derived target speed", "FWI starting model"),
    ):
        im = ax.imshow(image.T, cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        contour_mask(ax, brain, extent, "white")
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, label="m/s")
    savefig("fig02_acoustic_model")
    plt.close(fig)


def plot_reconstruction(result: dict) -> None:
    target = np.asarray(result["target_sound_speed_m_s"], dtype=float)
    recon = np.asarray(result["reconstruction_sound_speed_m_s"], dtype=float)
    enhanced = np.asarray(result["enhanced_reconstruction_sound_speed_m_s"], dtype=float)
    brain = np.asarray(result["brain_mask"], dtype=bool)
    extent = extent_mm(result)
    values = target[brain]
    vmin = np.percentile(values, 2)
    vmax = np.percentile(values, 98)
    err = np.where(brain, recon - target, np.nan)
    emax = np.nanpercentile(np.abs(err), 98)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), constrained_layout=True)
    for ax, image, title in zip(
        axes[:3],
        (target, recon, enhanced),
        ("Target brain speed", "FWI reconstruction", "Structure-enhanced FWI"),
    ):
        im = ax.imshow(image.T, cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        contour_mask(ax, brain, extent, "white")
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, label="m/s")

    im = axes[3].imshow(err.T, cmap="coolwarm", origin="lower", extent=extent, vmin=-emax, vmax=emax)
    axes[3].set_title("Reconstruction error")
    axes[3].set_xlabel("x [mm]")
    axes[3].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[3], label="m/s")
    savefig("fig03_brain_reconstruction")
    plt.close(fig)


def plot_ultrasound_data_reconstruction(result: dict) -> None:
    data = synthetic_data_tensor(result)
    migration = np.asarray(result["migration_sound_speed_m_s"], dtype=float)
    recon = np.asarray(result["reconstruction_sound_speed_m_s"], dtype=float)
    enhanced = np.asarray(result["enhanced_reconstruction_sound_speed_m_s"], dtype=float)
    target = np.asarray(result["target_sound_speed_m_s"], dtype=float)
    brain = np.asarray(result["brain_mask"], dtype=bool)
    extent = extent_mm(result)
    values = target[brain]
    vmin = np.percentile(values, 2)
    vmax = np.percentile(values, 98)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), constrained_layout=True)
    data_image = data.reshape(data.shape[0], -1).T
    im0 = axes[0].imshow(data_image, aspect="auto", cmap="magma")
    axes[0].set_title("Synthetic hemispherical-cap encoded data")
    axes[0].set_xlabel("source element")
    axes[0].set_ylabel("receiver-offset / frequency channel")
    fig.colorbar(im0, ax=axes[0], label="normalized phase data")

    for ax, image, title in zip(
        axes[1:],
        (migration, recon, enhanced),
        ("Adjoint migration reconstruction", "Iterative FWI reconstruction", "Enhanced display"),
    ):
        im = ax.imshow(image.T, cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        contour_mask(ax, brain, extent, "white")
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, label="m/s")

    savefig("fig05_simulated_ultrasound_reconstruction")
    plt.close(fig)


def plot_multislice_reconstruction(stack: list[tuple[dict, dict[str, float | int]]]) -> None:
    n_slices = len(stack)
    ct_slices = [np.asarray(result["ct_hu"], dtype=float) for result, _ in stack]
    targets = [np.asarray(result["target_sound_speed_m_s"], dtype=float) for result, _ in stack]
    recons = [np.asarray(result["reconstruction_sound_speed_m_s"], dtype=float) for result, _ in stack]
    masks = [np.asarray(result["brain_mask"], dtype=bool) for result, _ in stack]
    display_recons = [regularized_fwi_display(recon, mask) for recon, mask in zip(recons, masks)]
    skull_masks = [np.asarray(result["skull_mask"], dtype=bool) for result, _ in stack]
    ct_values = np.concatenate([ct[ct > -300.0] for ct in ct_slices])
    hu_min = np.percentile(ct_values, 1)
    hu_max = np.percentile(ct_values, 99)
    speed_values = np.concatenate([target[mask] for target, mask in zip(targets, masks)])
    vmin = np.percentile(speed_values, 2)
    vmax = np.percentile(speed_values, 98)
    errors = [
        np.where(mask, recon - target, np.nan)
        for target, recon, mask in zip(targets, display_recons, masks)
    ]
    emax = max(np.nanpercentile(np.abs(error), 98) for error in errors)

    fig, axes = plt.subplots(4, n_slices, figsize=(2.15 * n_slices, 9.1), constrained_layout=True)
    fig.suptitle(
        "3-D model-consistent synthetic hemispherical-cap FWI volume slices, not measured hardware data",
        fontsize=11,
    )
    if n_slices == 1:
        axes = axes.reshape(4, 1)

    ct_im = None
    speed_im = None
    error_im = None
    for col, ((result, metrics), ct, target, recon, error, mask, skull_mask) in enumerate(
        zip(stack, ct_slices, targets, display_recons, errors, masks, skull_masks)
    ):
        extent = extent_mm(result)
        slice_index = int(result["source_volume_index"])
        corr = float(metrics["pearson_correlation"])

        ax = axes[0, col]
        ct_im = ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=hu_min, vmax=hu_max)
        contour_mask(ax, skull_mask, extent, "yellow")
        contour_mask(ax, mask, extent, "cyan")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"z {slice_index}\nr={corr:.5f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("CT HU")

        for row, image, row_label in (
            (1, target, "CT-derived c target"),
            (2, recon, "regularized FWI"),
        ):
            ax = axes[row, col]
            speed_im = ax.imshow(image.T, cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
            contour_mask(ax, mask, extent, "white")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_label)

        ax = axes[3, col]
        error_im = ax.imshow(error.T, cmap="coolwarm", origin="lower", extent=extent, vmin=-emax, vmax=emax)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("error")

    if ct_im is not None:
        fig.colorbar(ct_im, ax=axes[0, :].ravel().tolist(), label="HU", shrink=0.82)
    if speed_im is not None:
        fig.colorbar(speed_im, ax=axes[1:3, :].ravel().tolist(), label="m/s", shrink=0.82)
    if error_im is not None:
        fig.colorbar(error_im, ax=axes[3, :].ravel().tolist(), label="m/s", shrink=0.82)
    savefig("fig06_multislice_reconstruction_stack")
    plt.close(fig)


def plot_optimization(result: dict) -> None:
    history = np.asarray(result["residual_history"], dtype=float)
    data = np.asarray(result["synthetic_data"], dtype=float)
    element_count = int(result["element_count"])
    channels = data.size // element_count

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    axes[0].semilogy(np.arange(history.size), history, marker="o", ms=3)
    axes[0].set_title("FWI objective")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("regularized objective")
    im = axes[1].imshow(data.reshape(element_count, channels).T, aspect="auto", cmap="magma")
    axes[1].set_title("Encoded finite-frequency data")
    axes[1].set_xlabel("source element")
    axes[1].set_ylabel("receiver-offset/frequency channel")
    fig.colorbar(im, ax=axes[1], label="normalized phase data")
    savefig("fig04_optimization_and_data")
    plt.close(fig)


def write_metrics(
    volume_result: dict,
    primary_slice: dict,
    metrics: dict[str, float | int],
    visible: bool,
    selected_iterations: int,
    stack: list[tuple[dict, dict[str, float | int]]],
    centroid_roi: dict,
) -> None:
    payload = {
        "ct_path": str(CT_PATH.relative_to(REPO_ROOT)),
        "grid_size": GRID_SIZE,
        "source_slice_index": int(volume_result["source_slice_index"]),
        "source_volume_index": int(primary_slice["source_volume_index"]),
        "volume_shape": [int(v) for v in np.asarray(volume_result["ct_hu"]).shape],
        "selected_iterations": selected_iterations,
        "visible_reconstruction": visible,
        "visibility_fraction_threshold": VISIBILITY_FRACTION,
        "minimum_objective_reduction": MIN_OBJECTIVE_REDUCTION,
        "element_count": int(volume_result["element_count"]),
        "frequencies_hz": [float(v) for v in volume_result["frequencies_hz"]],
        "receiver_offsets": [int(v) for v in volume_result["receiver_offsets"]],
        "inversion_dimensionality": str(volume_result["inversion_dimensionality"]),
        "geometry_model": str(volume_result["geometry_model"]),
        "operator_model": str(volume_result["operator_model"]),
        "slice_offset_m": float(primary_slice["slice_offset_m"]),
        "frequency_continuation": bool(volume_result["frequency_continuation"]),
        "sobolev_radius_voxels": int(volume_result["sobolev_radius_voxels"]),
        "sobolev_weight": float(volume_result["sobolev_weight"]),
        "enhancement_gain": float(volume_result["enhancement_gain"]),
        "edge_preserving_weight": float(volume_result["edge_preserving_weight"]),
        "edge_preserving_epsilon": float(volume_result["edge_preserving_epsilon"]),
        "edge_preserving_step": float(volume_result["edge_preserving_step"]),
        "edge_preserving_iterations": int(volume_result["edge_preserving_iterations"]),
        "figure06_display_regularization": "mask-aware diffusion plus clipped residual detail; no CT-target blending",
        "attenuation_model": bool(volume_result["attenuation_model"]),
        "nonlinear_harmonic_model": bool(volume_result["nonlinear_harmonic_model"]),
        "source_pressure_mpa": float(volume_result["source_pressure_mpa"]),
        "nonlinear_beta": float(volume_result["nonlinear_beta"]),
        "harmonic_count": int(volume_result["harmonic_count"]),
        "metrics": metrics,
        "multislice": {
            "slice_offsets": list(STACK_SLICE_OFFSETS),
            "slice_indices": [int(slice_result["source_volume_index"]) for slice_result, _ in stack],
            "metrics": [
                {
                    "source_volume_index": int(slice_result["source_volume_index"]),
                    "visible_reconstruction": visible_reconstruction(slice_metrics),
                    **slice_metrics,
                }
                for slice_result, slice_metrics in stack
            ],
        },
        "centroid_roi": centroid_roi,
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("  saved: docs/book/figures/ch27/metrics.json")


def run() -> dict[str, float | int]:
    if not CT_PATH.exists():
        raise FileNotFoundError(f"RIRE CT not found: {CT_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    volume_result, metrics, visible, selected_iterations = run_fwi_schedule()
    primary_slice = slice_volume_result(
        volume_result,
        int(volume_result["source_volume_index"]),
        metrics,
    )
    stack = run_multislice_stack(volume_result, metrics)
    plot_ct_and_geometry(primary_slice)
    plot_acoustic_model(primary_slice)
    plot_reconstruction(primary_slice)
    plot_optimization(primary_slice)
    plot_ultrasound_data_reconstruction(primary_slice)
    plot_multislice_reconstruction(stack)
    centroid_roi = plot_centroid_roi_stack(stack, CENTROID_ROI_HALF_WIDTH_MM, OUT_DIR)
    write_metrics(volume_result, primary_slice, metrics, visible, selected_iterations, stack, centroid_roi)
    if not visible:
        raise RuntimeError(
            "FWI reconstruction did not meet the chapter visibility criteria; "
            f"metrics={metrics}"
        )
    return metrics


if __name__ == "__main__" or __name__ == "ch27":
    run()
