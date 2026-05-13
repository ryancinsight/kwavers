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
        ct = np.asarray(result["placement_ct_hu"], dtype=float)
        spacing = tuple(float(v) for v in result["placement_spacing_m"])
        extent = image_extent_xy(ct, spacing)
        therapy_points = np.asarray(result["placement_therapy_points_m"], dtype=float)
        imaging_points = np.asarray(result["placement_imaging_points_m"], dtype=float)
        therapy_x = therapy_points[:, 0]
        therapy_y = therapy_points[:, 1]
        imaging_x = imaging_points[:, 0] if imaging_points.size else np.array([], dtype=float)
        imaging_y = imaging_points[:, 1] if imaging_points.size else np.array([], dtype=float)
        ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300)
        contour_mask(ax, np.asarray(result["placement_target_mask"], dtype=bool), extent, "yellow", 1.1)
        contour_mask(ax, np.asarray(result["placement_body_mask"], dtype=bool), extent, "cyan", 0.8)
        ax.scatter(therapy_x, therapy_y, s=2.0, c="#e74c3c", alpha=0.50, label="therapy tx/rx")
        if imaging_x.size > 0:
            ax.scatter(imaging_x, imaging_y, s=6.0, c="#2e86de", alpha=0.80, label="central imaging rx")
        focus = result["placement_focus_m"]
        skin = result["placement_skin_contact_m"]
        ax.scatter([focus[0]], [focus[1]], marker="x", s=45, c="white", linewidths=1.6)
        ax.scatter([skin[0]], [skin[1]], marker="o", s=24, c="lime", edgecolors="black", linewidths=0.5)
        ax.set_title(
            f"{result['anatomy']}: {short_device_name(result)}\n"
            f"full CT slice, {result['element_count']} elements, {placement_label(result)}"
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


def render_brain_helmet_3d(placement: dict[str, object]) -> Path:
    head = np.asarray(placement["head_surface_points_m"], dtype=float)
    skull = np.asarray(placement["skull_surface_points_m"], dtype=float)
    elements = np.asarray(placement["therapy_elements_m"], dtype=float)
    starts = np.asarray(placement["beam_start_points_m"], dtype=float)
    ends = np.asarray(placement["beam_end_points_m"], dtype=float)
    intersections = np.asarray(placement["skull_intersections_m"], dtype=float)
    focus = np.asarray(placement["focus_m"], dtype=float)

    fig = plt.figure(figsize=(13.5, 6.2), constrained_layout=True)
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (ax_a, ax_b):
        ax.scatter(head[:, 0], head[:, 1], head[:, 2], s=0.9, c="#b8c0cc", alpha=0.16, depthshade=False)
        ax.scatter(skull[:, 0], skull[:, 1], skull[:, 2], s=1.5, c="#f2d7a0", alpha=0.38, depthshade=False)
        ax.scatter(elements[:, 0], elements[:, 1], elements[:, 2], s=5.0, c="#d94f45", alpha=0.62, depthshade=False)
        if intersections.size:
            ax.scatter(
                intersections[:, 0],
                intersections[:, 1],
                intersections[:, 2],
                s=16,
                c="#ffff33",
                edgecolors="black",
                linewidths=0.25,
                depthshade=False,
            )
        ax.scatter([focus[0]], [focus[1]], [focus[2]], marker="x", s=80, c="white", linewidths=2.0)
        for start, end in zip(starts, ends):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                c="#ff8c00",
                lw=0.55,
                alpha=0.42,
            )
        set_equal_3d_limits(ax, [head, skull, elements])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
    ax_a.view_init(elev=18, azim=-68)
    ax_a.set_title(
        "1024-element helmet around CT head\n"
        "red: tx/rx elements, yellow: CT skull entry points",
        fontsize=10,
    )
    ax_b.view_init(elev=4, azim=-8)
    ax_b.set_title(
        "calvarial beam paths into brain target\n"
        f"skull-intersection fraction={float(placement['intersection_fraction']):.2f}",
        fontsize=10,
    )
    path = OUT_DIR / "fig03_brain_helmet_3d_placement.png"
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


def image_extent_xy(image: np.ndarray, spacing_m: tuple[float, float]) -> list[float]:
    nx, ny = image.shape
    return [
        -0.5 * (nx - 1) * spacing_m[0],
        0.5 * (nx - 1) * spacing_m[0],
        -0.5 * (ny - 1) * spacing_m[1],
        0.5 * (ny - 1) * spacing_m[1],
    ]


def short_device_name(result: dict[str, object]) -> str:
    name = str(result["device_model"])
    if "helmet" in name:
        return "INSIGHTEC-like helmet"
    return "HistoSonics-like skin arc"


def placement_label(result: dict[str, object]) -> str:
    metrics = result["placement_metrics"]
    gap_m = result.get(
        "placement_context_skin_gap_m",
        metrics["skin_contact_to_nearest_aperture_m"],
    )
    gap_mm = 1.0e3 * float(gap_m)
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


def set_equal_3d_limits(ax: plt.Axes, clouds: list[np.ndarray]) -> None:
    stacked = np.vstack([cloud for cloud in clouds if cloud.size])
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.52 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def write_metrics(
    results: list[dict[str, object]],
    figures: list[Path],
    brain_helmet_3d: dict[str, object],
) -> Path:
    payload = {
        "chapter": 29,
        "analysis": "same-device ultrasound treatment plus FWI/RTM monitoring",
        "simulation_type": "RITK-loaded CT/NIfTI, kwavers PyO3 theranostic FWI",
        "figures": [str(path) for path in figures],
        "brain_helmet_3d": {
            "geometry_model": brain_helmet_3d["geometry_model"],
            "element_count": int(brain_helmet_3d["element_count"]),
            "helmet_radius_m": float(brain_helmet_3d["helmet_radius_m"]),
            "beam_probe_count": int(np.asarray(brain_helmet_3d["beam_start_points_m"]).shape[0]),
            "skull_intersection_count": int(np.asarray(brain_helmet_3d["skull_intersections_m"]).shape[0]),
            "skull_intersection_fraction": float(brain_helmet_3d["intersection_fraction"]),
            "skull_hu_threshold": float(brain_helmet_3d["skull_hu_threshold"]),
        },
        "cases": [
            {
                "anatomy": result["anatomy"],
                "device_model": result["device_model"],
                "geometry_model": result["geometry_model"],
                "placement_context_model": result["placement_context_model"],
                "operator_model": result["operator_model"],
                "element_count": int(result["element_count"]),
                "source_pressure_pa": float(result["source_pressure_pa"]),
                "measurements": int(result["measurements"]),
                "active_voxels": int(result["active_voxels"]),
                "spacing_m": float(result["spacing_m"]),
                "placement_slice_index": int(result["placement_slice_index"]),
                "placement_spacing_m": [float(v) for v in result["placement_spacing_m"]],
                "placement_context_skin_gap_m": float(result["placement_context_skin_gap_m"]),
                "placement_context_surface_points": int(result["placement_context_surface_points"]),
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
    brain_helmet_3d = kw.plan_brain_helmet_placement_from_ritk_ct(
        str(CASES[0]["ct"]),
        element_count=int(CASES[0]["elements"]),
        surface_stride=int(os.environ.get("KWAVERS_CH29_3D_SURFACE_STRIDE", "7")),
        body_hu_threshold=float(os.environ.get("KWAVERS_CH29_3D_BODY_HU_THRESHOLD", "20.0")),
        skull_hu_threshold=float(os.environ.get("KWAVERS_CH29_3D_SKULL_HU_THRESHOLD", "300.0")),
    )
    figures = [
        render_layouts(results),
        render_reconstructions(results),
        render_brain_helmet_3d(brain_helmet_3d),
    ]
    metrics = write_metrics(results, figures, brain_helmet_3d)
    return {"figures": [str(path) for path in figures], "metrics": str(metrics)}


if __name__ == "__main__" or __name__ == "ch29":
    run()
