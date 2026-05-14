"""Chapter 28: CT-derived abdominal FWI for histotripsy analysis.

This script reuses the real kidney and liver CT/segmentation loaders from the
histotripsy chapters, runs a deterministic synthetic finite-frequency FWI
targeting reconstruction, and then runs reduced receiver inversions on
time-lapse lesion-state, Westervelt harmonic, and Rayleigh-Plesset subharmonic
source maps. The figures are model-consistent simulation outputs, not measured
transducer data or clinical targeting proof.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BOOK_DIR.parents[2]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch28"

sys.path.insert(0, str(BOOK_DIR))

import ch21d_real_kidney_ct_histotripsy as kidney  # noqa: E402
import ch21e_real_liver_ct_histotripsy as liver  # noqa: E402
from abdominal_fwi.model import (  # noqa: E402
    ApertureConfig,
    CaseResult,
    run_case,
)
from abdominal_fwi.constants import C_REF_M_S  # noqa: E402
from abdominal_fwi.metrics import as_metric_dict, equal_volume_detection  # noqa: E402
from abdominal_fwi.preprocessing import prepare_ct_slice  # noqa: E402


DX_M = float(os.environ.get("KWAVERS_CH28_FWI_DX_M", "0.002"))
GRID_SIZE = int(os.environ.get("KWAVERS_CH28_FWI_GRID_SIZE", "96"))
ELEMENT_COUNT = int(os.environ.get("KWAVERS_CH28_FWI_ELEMENTS", "256"))
ITERATIONS = int(os.environ.get("KWAVERS_CH28_FWI_ITERATIONS", "18"))


def load_cases() -> list[CaseResult]:
    """Load CT-derived kidney and liver slices, then run FWI for each."""

    config = ApertureConfig(element_count=ELEMENT_COUNT, iterations=ITERATIONS)
    cases = []

    ct, label, info = kidney.load_ct_and_segment(target_dx_m=DX_M)
    props = kidney.property_maps(label, f0=500_000.0)
    kidney_slice = prepare_ct_slice(
        name="kidney",
        title="KiTS19 kidney tumor",
        ct_hu=ct,
        label=label,
        sound_speed_m_s=props["c"],
        focus_index=int(info["focus_idx"][0]),
        input_spacing_m=float(info["dx"]),
        organ_labels=(kidney.KIDNEY.label,),
        target_labels=(kidney.TUMOR.label,),
        focus_indices=tuple(int(i) for i in info["focus_idx"]),
        slice_axis=int(np.argmin(label.shape)),
        output_size=GRID_SIZE,
    )
    cases.append(run_case(kidney_slice, config))

    ct, label, info = liver.load_ct_and_segment(target_dx_m=DX_M)
    props = liver.property_maps(label, f0=500_000.0)
    liver_slice = prepare_ct_slice(
        name="liver",
        title="LiTS liver HCC",
        ct_hu=ct,
        label=label,
        sound_speed_m_s=props["c"],
        focus_index=int(info["focus_idx"][0]),
        input_spacing_m=float(info["dx"]),
        organ_labels=(liver.LIVER.label,),
        target_labels=(liver.HCC.label,),
        focus_indices=tuple(int(i) for i in info["focus_idx"]),
        slice_axis=int(np.argmin(label.shape)),
        output_size=GRID_SIZE,
    )
    cases.append(run_case(liver_slice, config))
    return cases


def plot_case(result: CaseResult) -> Path:
    """Save a compact targeting and lesion-state reconstruction panel."""

    prepared = result.prepared
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), constrained_layout=True)
    fig.suptitle(
        (
            f"{prepared.title}: synthetic CT-derived abdominal FWI "
            "(not measured hardware data)"
        ),
        fontsize=11,
    )

    show_ct(axes[0, 0], prepared.ct_hu, prepared.organ_mask, prepared.target_mask)
    axes[0, 0].set_title("CT + labels")

    targeting_true = result.targeting.target_map * C_REF_M_S
    targeting_recon = result.targeting.reconstruction_map * C_REF_M_S
    targeting_error = result.targeting.error_map * C_REF_M_S
    vmax = robust_vmax(targeting_true, targeting_recon)
    show_field(axes[0, 1], targeting_true, "CT-derived anatomy", "m/s", vmax=vmax)
    show_field(axes[0, 2], targeting_recon, "FWI anatomy", "m/s", vmax=vmax)
    show_field(
        axes[0, 3],
        targeting_error,
        "targeting error",
        "m/s",
        vmax=max(float(np.max(np.abs(targeting_error))), 1.0),
    )

    lesion_true = result.lesioning.target_map * C_REF_M_S
    lesion_recon = result.lesioning.reconstruction_map * C_REF_M_S
    lesion_error = result.lesioning.error_map * C_REF_M_S
    lesion_vmax = max(abs(float(np.min(lesion_true))), 1.0)
    show_field(axes[1, 0], lesion_true, "lesion delta c", "m/s", vmax=lesion_vmax)
    show_field(axes[1, 1], lesion_recon, "time-lapse FWI", "m/s", vmax=lesion_vmax)
    show_detection(
        axes[1, 2],
        result,
        result.lesioning,
        "lesion detection",
        negative=True,
    )
    show_objective(
        axes[1, 3],
        (
            ("targeting", result.targeting),
            ("lesioning", result.lesioning),
        ),
    )

    output = OUT_DIR / f"fig01_{prepared.name}_abdominal_fwi.png"
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return output


def plot_advanced_case(result: CaseResult) -> Path:
    """Save subharmonic and nonlinear reconstruction panels."""

    prepared = result.prepared
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), constrained_layout=True)
    fig.suptitle(
        (
            f"{prepared.title}: subharmonic and nonlinear abdominal FWI "
            "(synthetic model-consistent channels)"
        ),
        fontsize=11,
    )

    show_ct(axes[0, 0], prepared.ct_hu, prepared.organ_mask, prepared.target_mask)
    axes[0, 0].set_title("CT + labels")
    show_field(
        axes[0, 1],
        result.subharmonic.target_map,
        "subharmonic source",
        "a.u.",
        vmax=1.0,
    )
    show_field(
        axes[0, 2],
        result.subharmonic.reconstruction_map,
        "subharmonic FWI",
        "a.u.",
        vmax=robust_vmax(result.subharmonic.reconstruction_map),
    )
    show_detection(
        axes[0, 3],
        result,
        result.subharmonic,
        "subharmonic detection",
        negative=False,
    )

    nonlinear_true = result.nonlinear.target_map * C_REF_M_S
    nonlinear_recon = result.nonlinear.reconstruction_map * C_REF_M_S
    nonlinear_vmax = max(
        float(np.max(np.abs(nonlinear_true))),
        robust_vmax(nonlinear_recon),
    )
    show_field(
        axes[1, 0],
        nonlinear_true,
        "nonlinear susceptibility",
        "m/s eq.",
        vmax=nonlinear_vmax,
    )
    show_field(
        axes[1, 1],
        nonlinear_recon,
        "nonlinear FWI",
        "m/s eq.",
        vmax=nonlinear_vmax,
    )
    show_detection(
        axes[1, 2],
        result,
        result.nonlinear,
        "nonlinear detection",
        negative=False,
    )
    show_objective(
        axes[1, 3],
        (
            ("subharmonic", result.subharmonic),
            ("nonlinear", result.nonlinear),
        ),
    )

    output = OUT_DIR / f"fig02_{prepared.name}_subharmonic_nonlinear_fwi.png"
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return output


def show_ct(ax: plt.Axes, ct_hu: np.ndarray, organ: np.ndarray, target: np.ndarray) -> None:
    """Draw CT HU with organ and target contours."""

    ax.imshow(ct_hu.T, cmap="gray", origin="lower", vmin=-150, vmax=250)
    ax.contour(organ.T, levels=[0.5], colors=["cyan"], linewidths=1.0)
    ax.contour(target.T, levels=[0.5], colors=["yellow"], linewidths=1.4)
    ax.set_xticks([])
    ax.set_yticks([])


def show_field(
    ax: plt.Axes,
    field: np.ndarray,
    title: str,
    label: str,
    *,
    vmax: float,
) -> None:
    """Draw a signed acoustic-property field."""

    im = ax.imshow(field.T, cmap="coolwarm", origin="lower", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)


def show_detection(
    ax: plt.Axes,
    result: CaseResult,
    inversion,
    title: str,
    *,
    negative: bool,
) -> None:
    """Overlay equal-volume detections on the CT-derived target."""

    active = result.prepared.imaging_mask
    detection_active = equal_volume_detection(
        inversion.reconstruction_map[active],
        result.lesion_mask[active],
        negative=negative,
    )
    detected = np.zeros_like(result.lesion_mask, dtype=bool)
    detected[active] = detection_active

    ax.imshow(result.prepared.ct_hu.T, cmap="gray", origin="lower", vmin=-150, vmax=250)
    ax.contour(result.lesion_mask.T, levels=[0.5], colors=["lime"], linewidths=1.4)
    ax.contour(detected.T, levels=[0.5], colors=["red"], linewidths=1.2)
    ax.set_title(
        f"{title}\n"
        f"Dice={inversion.metrics['equal_volume_dice']:.3f}"
    )
    ax.set_xticks([])
    ax.set_yticks([])


def show_objective(ax: plt.Axes, entries: tuple[tuple[str, object], ...]) -> None:
    """Plot normalized objective histories."""

    for label, inversion in entries:
        history = np.asarray(inversion.metrics["objective_history"], dtype=float)
        history = history / max(float(history[0]), 1.0e-30)
        ax.plot(np.arange(history.size), history, marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("PCG iteration")
    ax.set_ylabel("normalized objective")
    ax.set_title("solver convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def robust_vmax(*fields: np.ndarray) -> float:
    """Return a shared display scale that preserves soft-tissue structure."""

    values = np.concatenate([np.abs(field).ravel() for field in fields])
    values = values[values > 0.0]
    if values.size == 0:
        return 1.0
    return min(max(float(np.percentile(values, 97.0)), 1.0), 220.0)


def write_metrics(results: list[CaseResult]) -> Path:
    """Write reproducible scalar diagnostics for the generated figures."""

    payload = {
        "chapter": 28,
        "analysis": "CT-derived abdominal histotripsy FWI targeting and lesioning",
        "simulation_type": (
            "synthetic CT-derived Born inversion with bounded 2-D Westervelt "
            "and Rayleigh-Plesset source maps"
        ),
        "dx_m": DX_M,
        "grid_size": GRID_SIZE,
        "element_count": ELEMENT_COUNT,
        "iterations": ITERATIONS,
        "cases": [as_metric_dict(result) for result in results],
    }
    output = OUT_DIR / "metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def run() -> dict[str, object]:
    """Generate Chapter 28 abdominal FWI figures and metrics."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_cases()
    figure_paths = []
    for result in results:
        figure_paths.append(plot_case(result))
        figure_paths.append(plot_advanced_case(result))
    metrics_path = write_metrics(results)
    return {
        "figures": [str(path) for path in figure_paths],
        "metrics": str(metrics_path),
        "cases": [as_metric_dict(result) for result in results],
    }


if __name__ == "__main__" or __name__ == "ch28":
    run()
