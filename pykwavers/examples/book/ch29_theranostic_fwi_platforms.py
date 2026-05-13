"""Chapter 29: same-device therapy and FWI/RTM monitoring simulations.

The computation is owned by kwavers through the PyO3 wrapper
``run_theranostic_fwi_from_ritk``. Python only selects the public CT/NIfTI
inputs, runs the wrapper, and renders figures.
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
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch29"
PY_PACKAGE = REPO_ROOT / "pykwavers" / "python"

if "PYKWAVERS_EXTENSION_PATH" not in os.environ:
    for candidate in (
        REPO_ROOT / "target" / "release" / "pykwavers.dll",
        REPO_ROOT / "target" / "maturin" / "pykwavers.dll",
        REPO_ROOT / "target" / "debug" / "pykwavers.dll",
    ):
        if candidate.exists():
            os.environ["PYKWAVERS_EXTENSION_PATH"] = str(candidate)
            break
if str(PY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PY_PACKAGE))

import pykwavers as kw  # noqa: E402


CASES = (
    {
        "name": "brain",
        "title": "Brain helmet",
        "ct": REPO_ROOT / "data" / "rire_patient_109" / "patient_109_ct.nii.gz",
        "seg": None,
        "grid": int(os.environ.get("KWAVERS_CH29_BRAIN_GRID", "48")),
        "elements": 1024,
        "freq": [220_000.0, 650_000.0],
        "offsets": [256, 384, 512, 640],
        "pressure": 1.5e5,
    },
    {
        "name": "kidney",
        "title": "Kidney histotripsy head",
        "ct": REPO_ROOT / "data" / "kits19_sample" / "case_00000.nii.gz",
        "seg": REPO_ROOT / "data" / "kits19_sample" / "segmentation_00000.nii.gz",
        "grid": int(os.environ.get("KWAVERS_CH29_ABDOMEN_GRID", "52")),
        "elements": 256,
        "freq": [250_000.0, 500_000.0, 750_000.0],
        "offsets": [32, 64, 96, 128],
        "pressure": 28.0e6,
    },
    {
        "name": "liver",
        "title": "Liver histotripsy head",
        "ct": REPO_ROOT / "data" / "lits17_sample" / "volume-0.nii",
        "seg": REPO_ROOT / "data" / "lits17_sample" / "segmentation-0.nii",
        "grid": int(os.environ.get("KWAVERS_CH29_ABDOMEN_GRID", "52")),
        "elements": 256,
        "freq": [250_000.0, 500_000.0, 750_000.0],
        "offsets": [32, 64, 96, 128],
        "pressure": 28.0e6,
    },
)


def run_case(case: dict[str, object]) -> dict[str, object]:
    if not Path(case["ct"]).exists():
        raise FileNotFoundError(case["ct"])
    seg = case["seg"]
    if seg is not None and not Path(seg).exists():
        raise FileNotFoundError(seg)
    return kw.run_theranostic_fwi_from_ritk(
        str(case["ct"]),
        None if seg is None else str(seg),
        anatomy=str(case["name"]),
        grid_size=int(case["grid"]),
        element_count=int(case["elements"]),
        iterations=int(os.environ.get("KWAVERS_CH29_ITERATIONS", "10")),
        frequencies_hz=list(case["freq"]),
        receiver_offsets=list(case["offsets"]),
        source_pressure_pa=float(case["pressure"]),
        noise_fraction=float(os.environ.get("KWAVERS_CH29_NOISE_FRACTION", "0.012")),
    )


def render_layouts(results: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.8), constrained_layout=True)
    for ax, result in zip(axes, results):
        ct = np.asarray(result["ct_hu"], dtype=float)
        dx = float(result["spacing_m"])
        extent = image_extent(ct, dx)
        therapy_x = np.asarray(result["therapy_x_m"], dtype=float)
        therapy_y = np.asarray(result["therapy_y_m"], dtype=float)
        imaging_x = np.asarray(result["imaging_x_m"], dtype=float)
        imaging_y = np.asarray(result["imaging_y_m"], dtype=float)
        ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300)
        contour_mask(ax, np.asarray(result["target_mask"], dtype=bool), extent, "yellow", 1.1)
        contour_mask(ax, np.asarray(result["body_mask"], dtype=bool), extent, "cyan", 0.8)
        ax.scatter(therapy_x, therapy_y, s=2.0, c="#e74c3c", alpha=0.50, label="therapy tx/rx")
        if imaging_x.size > 0:
            ax.scatter(imaging_x, imaging_y, s=6.0, c="#2e86de", alpha=0.80, label="central imaging rx")
        focus = result["focus_m"]
        skin = result["skin_contact_m"]
        ax.scatter([focus[0]], [focus[1]], marker="x", s=45, c="white", linewidths=1.6)
        ax.scatter([skin[0]], [skin[1]], marker="o", s=24, c="lime", edgecolors="black", linewidths=0.5)
        ax.set_title(
            f"{result['anatomy']}: {short_device_name(result)}\n"
            f"{result['element_count']} elements, {placement_label(result)}"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_xlim(*axis_limits(extent[:2], therapy_x, imaging_x))
        ax.set_ylim(*axis_limits(extent[2:], therapy_y, imaging_y))
        ax.legend(loc="lower right", fontsize=7, frameon=True)
    path = OUT_DIR / "fig01_device_placement_on_ct.png"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def render_reconstructions(results: list[dict[str, object]]) -> Path:
    columns = (
        ("exposure", "magma", "simulated exposure"),
        ("lesion_target", "magma", "lesion target"),
        ("active_lesion_reconstruction", "viridis", "active FWI"),
        ("subharmonic_reconstruction", "viridis", "subharmonic RTM/FWI"),
        ("harmonic_reconstruction", "viridis", "harmonic FWI"),
        ("ultraharmonic_reconstruction", "viridis", "ultraharmonic FWI"),
        ("fused_reconstruction", "viridis", "fusion"),
    )
    fig, axes = plt.subplots(len(results), len(columns), figsize=(18.5, 8.4), constrained_layout=True)
    for row, result in enumerate(results):
        for col, (key, cmap, title) in enumerate(columns):
            ax = axes[row, col]
            image = np.asarray(result[key], dtype=float)
            im = ax.imshow(image.T, cmap=cmap, origin="lower", vmin=0.0, vmax=max(float(np.max(image)), 1.0e-12))
            ax.contour(np.asarray(result["target_mask"], dtype=bool).T, levels=[0.5], colors="white", linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{result['anatomy']} {title}" if col == 0 else title, fontsize=9)
            if col == len(columns) - 1:
                metrics = result["metrics"]["fusion"]
                ax.set_xlabel(f"Dice={metrics['dice_equal_area']:.2f}, CNR={metrics['cnr']:.2f}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    path = OUT_DIR / "fig02_exposure_and_reconstruction.png"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def image_extent(image: np.ndarray, spacing_m: float) -> list[float]:
    nx, ny = image.shape
    return [
        -0.5 * (nx - 1) * spacing_m,
        0.5 * (nx - 1) * spacing_m,
        -0.5 * (ny - 1) * spacing_m,
        0.5 * (ny - 1) * spacing_m,
    ]


def short_device_name(result: dict[str, object]) -> str:
    name = str(result["device_model"])
    if "helmet" in name:
        return "INSIGHTEC-like helmet"
    return "HistoSonics-like skin arc"


def placement_label(result: dict[str, object]) -> str:
    metrics = result["placement_metrics"]
    gap_mm = 1.0e3 * float(metrics["skin_contact_to_nearest_aperture_m"])
    clearance_mm = 1.0e3 * float(metrics["min_body_clearance_m"])
    if str(result["anatomy"]) == "brain":
        return f"helmet clearance {clearance_mm:.1f} mm"
    return f"skin gap {gap_mm:.1f} mm"


def axis_limits(
    image_limits: list[float] | tuple[float, float],
    therapy_values: np.ndarray,
    imaging_values: np.ndarray,
) -> tuple[float, float]:
    values = [float(image_limits[0]), float(image_limits[1])]
    if therapy_values.size > 0:
        values.extend([float(np.min(therapy_values)), float(np.max(therapy_values))])
    if imaging_values.size > 0:
        values.extend([float(np.min(imaging_values)), float(np.max(imaging_values))])
    low = min(values)
    high = max(values)
    margin = max(0.04 * (high - low), 5.0e-3)
    return low - margin, high + margin


def contour_mask(ax: plt.Axes, mask: np.ndarray, extent: list[float], color: str, width: float) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors=color, linewidths=width)


def write_metrics(results: list[dict[str, object]], figures: list[Path]) -> Path:
    payload = {
        "chapter": 29,
        "analysis": "same-device ultrasound treatment plus FWI/RTM monitoring",
        "simulation_type": "RITK-loaded CT/NIfTI, kwavers PyO3 theranostic FWI",
        "figures": [str(path) for path in figures],
        "cases": [
            {
                "anatomy": result["anatomy"],
                "device_model": result["device_model"],
                "geometry_model": result["geometry_model"],
                "operator_model": result["operator_model"],
                "element_count": int(result["element_count"]),
                "source_pressure_pa": float(result["source_pressure_pa"]),
                "measurements": int(result["measurements"]),
                "active_voxels": int(result["active_voxels"]),
                "spacing_m": float(result["spacing_m"]),
                "placement_metrics": result["placement_metrics"],
                "metrics": result["metrics"],
            }
            for result in results
        ],
    }
    path = OUT_DIR / "metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_case(case) for case in CASES]
    figures = [render_layouts(results), render_reconstructions(results)]
    metrics = write_metrics(results, figures)
    return {"figures": [str(path) for path in figures], "metrics": str(metrics)}


if __name__ == "__main__" or __name__ == "ch29":
    run()
