from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ReportConfig:
    out_dir: Path
    repo_root: Path
    ct_path: Path
    grid_size: int
    element_count: int
    active_frequencies_hz: tuple[float, ...]
    passive_frequencies_hz: tuple[float, ...]
    receiver_offsets: tuple[int, ...]
    noise_snr_db: float
    gain_jitter_std: float
    phase_jitter_rad: float


def savefig(name: str, report: ReportConfig) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(report.out_dir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch27/{name}.{{pdf,png}}")


def image_extent_mm(baseline: dict) -> tuple[float, float, float, float]:
    ct = np.asarray(baseline["ct_hu"])
    spacing_mm = 1.0e3 * float(baseline["spacing_m"])
    return (-0.5 * ct.shape[0] * spacing_mm, 0.5 * ct.shape[0] * spacing_mm, -0.5 * ct.shape[1] * spacing_mm, 0.5 * ct.shape[1] * spacing_mm)


def contour_mask(ax: plt.Axes, mask: np.ndarray, extent: tuple[float, float, float, float], color: str, level: float) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[level], colors=[color], linewidths=0.55)


def plot_scenarios(results: list[Any], baseline: dict, report: ReportConfig) -> None:
    ct = np.asarray(baseline["ct_hu"], dtype=float)
    mask = np.asarray(baseline["brain_mask"], dtype=bool)
    extent = image_extent_mm(baseline)
    columns = [("CT HU", "gray"), ("lesion target", "magma"), ("normal FWI", "viridis"), ("multiparam FWI", "viridis"), ("nonlinear FWI", "viridis"), ("subharmonic FWI", "viridis"), ("fused", "viridis")]
    fig, axes = plt.subplots(len(results), len(columns), figsize=(15.5, 3.0 * len(results)), constrained_layout=True)
    for row, result in enumerate(results):
        images = [ct, result.lesion, result.linear_fwi, result.multiparameter_fwi, result.nonlinear_fwi, result.subharmonic_source, result.fused]
        for col, (title, cmap) in enumerate(columns):
            ax = axes[row, col]
            im = ax.imshow(images[col].T, origin="lower", extent=extent, cmap=cmap, vmin=-150 if title == "CT HU" else 0.0, vmax=900 if title == "CT HU" else 1.0)
            contour_mask(ax, mask, extent, "white", 0.45)
            if col == 0:
                ax.set_ylabel(result.scenario.title)
            if row == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == len(columns) - 1:
                metric = result.metrics["fused"]
                ax.text(0.02, 0.04, f"Dice {metric['dice_equal_area']:.2f}\nAUPRC {metric['auprc']:.2f}", transform=ax.transAxes, color="white")
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), label="normalized lesion score", shrink=0.72)
    savefig("fig08_histotripsy_custom_reconstruction_scenarios", report)
    plt.close(fig)


def plot_metrics(results: list[Any], report: ReportConfig) -> None:
    methods = ["linear_fwi", "multiparameter_fwi", "nonlinear_fwi", "subharmonic_source", "fused"]
    labels = ["normal", "multiparam", "nonlinear", "subharmonic", "fused"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    x = np.arange(len(results))
    width = 0.15
    for idx, (method, label) in enumerate(zip(methods, labels)):
        offset = (idx - 0.5 * (len(methods) - 1)) * width
        axes[0].bar(x + offset, [r.metrics[method]["dice_equal_area"] for r in results], width, label=label)
        axes[1].bar(x + offset, [r.metrics[method]["auprc"] for r in results], width, label=label)
        axes[2].bar(x + offset, [r.metrics[method]["cnr"] for r in results], width, label=label)
    for ax, title in zip(axes, ("equal-area Dice", "AUPRC", "CNR")):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([r.scenario.name for r in results], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[2].legend(loc="upper left", ncols=1)
    savefig("fig09_histotripsy_reconstruction_metrics", report)
    plt.close(fig)


def plot_passive_bands(result: Any, baseline: dict, report: ReportConfig) -> None:
    extent = image_extent_mm(baseline)
    mask = np.asarray(baseline["brain_mask"], dtype=bool)
    images = [result.lesion, result.linear_fwi, result.multiparameter_fwi, result.nonlinear_fwi, *result.passive_bands, result.subharmonic_source, result.fused]
    titles = ["target", "normal FWI", "multiparam", "nonlinear", *[f"{f/1e3:.0f} kHz" for f in report.passive_frequencies_hz], "subharmonic", "fusion"]
    fig, axes = plt.subplots(1, len(images), figsize=(2.05 * len(images), 2.7), constrained_layout=True)
    for ax, image, title in zip(axes, images, titles):
        im = ax.imshow(image.T, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
        contour_mask(ax, mask, extent, "white", 0.45)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), label="normalized score", shrink=0.75)
    savefig("fig10_histotripsy_passive_band_rtm", report)
    plt.close(fig)


def write_metrics(results: list[Any], baseline: dict, report: ReportConfig) -> None:
    payload = {
        "ct_path": str(report.ct_path.relative_to(report.repo_root)),
        "grid_size": report.grid_size,
        "element_count": report.element_count,
        "active_frequencies_hz": list(report.active_frequencies_hz),
        "passive_frequencies_hz": list(report.passive_frequencies_hz),
        "receiver_offsets": list(report.receiver_offsets),
        "noise_snr_db": report.noise_snr_db,
        "gain_jitter_std": report.gain_jitter_std,
        "phase_jitter_rad": report.phase_jitter_rad,
        "robust_misfit": "Huber IRLS",
        "source_slice_index": int(baseline["source_slice_index"]),
        "scenario_metrics": {result.scenario.name: result.metrics for result in results},
        "objective_history": {result.scenario.name: {"linear": result.objective_linear, "multiparameter": result.objective_multiparameter, "nonlinear": result.objective_nonlinear, "subharmonic": result.objective_subharmonic} for result in results},
    }
    with open(report.out_dir / "histotripsy_monitoring_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("  saved: docs/book/figures/ch27/histotripsy_monitoring_metrics.json")
