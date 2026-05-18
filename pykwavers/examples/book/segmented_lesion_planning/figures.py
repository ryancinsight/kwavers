"""Figure and metrics writers for Chapter 32."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .types import SegmentationGrid, Tissue


LABEL_CMAP = ListedColormap(
    [
        "#101820",
        "#d7c9a8",
        "#f5d46b",
        "#d9d9d9",
        "#d94f45",
        "#24a6d8",
    ]
)


def render_plan(grid: SegmentationGrid, result: dict[str, object], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_segmentation_and_scores(grid, result, out_dir / "fig01_segmentation_candidate_scores.png"),
        plot_optimized_field(grid, result, out_dir / "fig02_optimized_spot_and_avoidance.png"),
        plot_solver_components(result, out_dir / "fig03_hybrid_solver_tradeoffs.png"),
    ]
    for path in paths:
        save_pdf_twin(path)
    return paths


def plot_segmentation_and_scores(
    grid: SegmentationGrid,
    result: dict[str, object],
    path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    extent = extent_mm(grid)
    axes[0].imshow(grid.labels.T, origin="lower", extent=extent, cmap=LABEL_CMAP, vmin=0, vmax=5)
    contour_masks(axes[0], grid, extent)
    best = result["best"]
    single_candidates = [
        candidate for candidate in result["candidates"]
        if len(candidate["aperture"].source_angles_deg) <= 1
    ]
    for candidate in single_candidates:
        aperture = candidate["aperture"]
        center = aperture.center_m * 1.0e3
        color = "#2f2f2f" if aperture.angle_deg != best["aperture"].angle_deg else "#ff006e"
        axes[0].scatter(center[0], center[1], s=22, color=color)
    draw_aperture(axes[0], best["aperture"], "#ff006e")
    axes[0].set_title("Segmentation and aperture candidates")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    angles = [candidate["aperture"].angle_deg for candidate in single_candidates]
    scores = [candidate["metrics"]["score"] for candidate in single_candidates]
    bone = [candidate["metrics"]["bone_path_fraction"] for candidate in single_candidates]
    air = [candidate["metrics"]["air_path_fraction"] for candidate in single_candidates]
    axes[1].plot(angles, scores, marker="o", color="#2d6cdf", label="objective")
    axes[1].set_xlabel("candidate angle [deg]")
    axes[1].set_ylabel("objective")
    ax2 = axes[1].twinx()
    ax2.plot(angles, bone, marker="s", color="#8a8a8a", label="bone path")
    ax2.plot(angles, air, marker="^", color="#101820", label="air path")
    ax2.set_ylabel("path fraction")
    axes[1].axvline(best["aperture"].angle_deg, color="#ff006e", linewidth=1.2)
    axes[1].grid(alpha=0.25)
    lines = axes[1].lines + ax2.lines
    axes[1].legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    axes[1].set_title("Hybrid access score")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_optimized_field(
    grid: SegmentationGrid,
    result: dict[str, object],
    path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    extent = extent_mm(grid)
    axes[0].imshow(grid.labels.T, origin="lower", extent=extent, cmap=LABEL_CMAP, vmin=0, vmax=5, alpha=0.92)
    contour_masks(axes[0], grid, extent)
    draw_aperture(axes[0], result["best"]["aperture"], "#ff006e")
    focus = np.asarray(result["target_centroid_m"]) * 1.0e3
    for element in result["best"]["aperture"].element_positions_m[::2]:
        axes[0].plot([element[0] * 1.0e3, focus[0]], [element[1] * 1.0e3, focus[1]], color="#ff006e", alpha=0.28)
    axes[0].scatter([focus[0]], [focus[1]], marker="x", s=70, color="white", linewidths=1.8)
    axes[0].set_title("Selected path avoids segmented barriers")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    intensity = np.asarray(result["normalized_intensity"], dtype=float)
    body_intensity = np.ma.masked_where(~grid.body_mask, np.clip(intensity, 0.0, 1.2))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#000000")
    image = axes[1].imshow(
        body_intensity.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=1.2,
    )
    contour_masks(axes[1], grid, extent)
    axes[1].set_title("Body-masked optimized lesioning field")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    fig.colorbar(image, ax=axes[1], shrink=0.82, label="intensity / tumor peak")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_solver_components(result: dict[str, object], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    best = result["best"]
    drive = np.asarray(best["drive_weights"])
    index = np.arange(drive.size)
    axes[0].bar(index - 0.18, np.abs(drive), width=0.36, color="#2d6cdf", label="amplitude")
    axes[0].bar(index + 0.18, np.angle(drive) / np.pi, width=0.36, color="#f59e0b", label="phase / pi")
    axes[0].set_xlabel("element index")
    axes[0].set_title("Complex drive solved by ridge LS")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = [
        "tumor mean",
        "coverage",
        "protected peak",
        "normal mean",
        "sidelobe peak",
        "sidelobe p99",
        "air path",
        "bone path",
        "fat path",
    ]
    summary = result["summary"]
    values = [
        summary["tumor_mean_intensity"],
        summary["tumor_coverage_fraction"],
        summary["protected_peak_ratio"],
        summary["normal_mean_ratio"],
        summary["body_sidelobe_peak_ratio"],
        summary["body_sidelobe_p99_ratio"],
        summary["air_path_fraction"],
        summary["bone_path_fraction"],
        summary["fat_path_fraction"],
    ]
    colors = [
        "#2a9d8f",
        "#2a9d8f",
        "#d94f45",
        "#8a8a8a",
        "#c77dff",
        "#9d4edd",
        "#101820",
        "#6f6f6f",
        "#f5d46b",
    ]
    axes[1].barh(labels, values, color=colors)
    axes[1].set_xlim(0.0, max(1.0, float(np.max(values)) * 1.1))
    axes[1].set_title("Acceptance metrics")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_metrics(
    grid: SegmentationGrid,
    result: dict[str, object],
    figures: list[Path],
    path: Path,
) -> Path:
    best = result["best"]
    payload = {
        "chapter": 32,
        "analysis": "segmentation-driven transducer placement and focal spot shaping",
        "dataset": result.get("dataset", {"source": "analytic segmented phantom"}),
        "segmentation_voxels": {
            tissue.name.lower(): int(np.count_nonzero(grid.mask(tissue)))
            for tissue in Tissue
        },
        "config": asdict(result["config"]),
        "selected_aperture": {
            "angle_deg": float(best["aperture"].angle_deg),
            "source_angles_deg": list(best["aperture"].source_angles_deg),
            "center_m": best["aperture"].center_m.tolist(),
            "element_count": int(best["aperture"].element_positions_m.shape[0]),
        },
        "summary": result["summary"],
        "candidate_metrics": [
            {
                "angle_deg": float(candidate["aperture"].angle_deg),
                **candidate["metrics"],
            }
            for candidate in result["candidates"]
        ],
        "figures": [str(item) for item in figures],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def extent_mm(grid: SegmentationGrid) -> list[float]:
    nx, ny = grid.shape
    half_x = 0.5 * (nx - 1) * grid.spacing_m * 1.0e3
    half_y = 0.5 * (ny - 1) * grid.spacing_m * 1.0e3
    return [-half_x, half_x, -half_y, half_y]


def contour_masks(ax: plt.Axes, grid: SegmentationGrid, extent: list[float]) -> None:
    contours = [
        (grid.body_mask, "#ffffff", 0.8),
        (grid.mask(Tissue.TUMOR), "#ffef5e", 1.4),
        (grid.mask(Tissue.AVOID), "#00c8ff", 1.3),
        (grid.mask(Tissue.BONE), "#4a4a4a", 0.8),
    ]
    for mask, color, linewidth in contours:
        if np.any(mask):
            ax.contour(mask.T.astype(float), levels=[0.5], colors=[color], linewidths=linewidth, extent=extent)


def draw_aperture(ax: plt.Axes, aperture, color: str) -> None:
    elements = aperture.element_positions_m * 1.0e3
    ax.scatter(elements[:, 0], elements[:, 1], s=14, color=color, zorder=5)
    ax.scatter([aperture.center_m[0] * 1.0e3], [aperture.center_m[1] * 1.0e3], s=42, color=color, marker="D")


def save_pdf_twin(path: Path) -> None:
    png = plt.imread(path)
    fig, ax = plt.subplots(figsize=(8, 8 * png.shape[0] / png.shape[1]))
    ax.imshow(png)
    ax.axis("off")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
