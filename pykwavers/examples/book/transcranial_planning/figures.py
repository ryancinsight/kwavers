from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import BrainTriplet, DatasetSource, FIG_DIR, normalize_unit
from .registration import AffineRegistrationResult, RegistrationResult
from .simulation import AcousticResult, BbbOpeningResult, SubspotPlan, ThermalResult
from .transducer import PhaseCorrection


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = path.with_suffix(f".{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  saved: {out.relative_to(FIG_DIR.parents[3])}")


def write_json(name: str, payload: dict) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  saved: {path.relative_to(FIG_DIR.parents[3])}")
    return path


def dataset_manifest(sources: list[DatasetSource]) -> dict:
    return {
        "sources": [
            {
                "name": src.name,
                "role": src.role,
                "path": str(src.path) if src.path is not None else None,
                "source_url": src.source_url,
                "license": src.license,
                "present": src.present,
            }
            for src in sources
        ]
    }


def rot90_xy(index_2d: tuple[int, int], plane_shape: tuple[int, int]) -> tuple[int, int]:
    return index_2d[0], plane_shape[1] - 1 - index_2d[1]


def target_plane(
    data: np.ndarray,
    index: tuple[int, int, int],
    plane: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    if plane == "axial":
        return data[:, :, index[2]], (index[0], index[1])
    if plane == "coronal":
        return data[:, index[1], :], (index[0], index[2])
    if plane == "sagittal":
        return data[index[0], :, :], (index[1], index[2])
    raise ValueError(f"Unknown plane: {plane}")


def plot_registration_inputs(triplet: BrainTriplet, registration: RegistrationResult) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.2))
    panels = [
        ("CT skull", normalize_unit(triplet.ct_hu.data)),
        ("T1 MRI in CT space", normalize_unit(registration.t1_registered.data)),
        ("MNI atlas in CT space", normalize_unit(registration.atlas_registered.data)),
    ]
    planes = [("axial", "Axial"), ("coronal", "Coronal"), ("sagittal", "Sagittal")]
    for row, (plane, plane_title) in enumerate(planes):
        skull_plane, _ = target_plane(triplet.skull_mask, triplet.target_index, plane)
        skull_slice = np.rot90(skull_plane)
        for col, (title, data) in enumerate(panels):
            ax = axes[row, col]
            plane_data, index_2d = target_plane(data, triplet.target_index, plane)
            ax.imshow(np.rot90(plane_data), cmap="gray", origin="lower")
            if np.any(skull_slice):
                ax.contour(skull_slice, levels=[0.5], colors=["#d95f02"], linewidths=0.7)
            x, y = rot90_xy(index_2d, plane_data.shape)
            ax.plot(x, y, "r+", ms=8, mew=1.4)
            ax.set_title(f"{title}\n{plane_title}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(registration.message)
    fig.tight_layout()
    savefig(FIG_DIR / "fig01_registered_ct_mri_mni")
    plt.close(fig)


def plot_affine_registration_qc(result: AffineRegistrationResult, tumor: np.ndarray | None = None) -> None:
    z = result.fixed.data.shape[2] // 2
    if tumor is not None and np.any(tumor):
        z = int(np.rint(np.argwhere(tumor).mean(axis=0)[2]))
    fixed = normalize_unit(result.fixed.data)
    moving = normalize_unit(result.moving_registered.data)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    panels = [
        ("MRI fixed", fixed[:, :, z], "gray"),
        ("registered CT", moving[:, :, z], "gray"),
        ("CT/MRI overlay", fixed[:, :, z], "gray"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        ax.imshow(np.rot90(data), cmap=cmap, origin="lower")
        if title == "CT/MRI overlay":
            ax.contour(np.rot90(moving[:, :, z]), levels=[0.5], colors=["#d95f02"], linewidths=0.7)
        if tumor is not None and np.any(tumor[:, :, z]):
            ax.contour(np.rot90(tumor[:, :, z]), levels=[0.5], colors=["cyan"], linewidths=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{result.method}: NMI={result.nmi:.3f}, edge overlap={result.edge_overlap:.3f}")
    fig.tight_layout()
    savefig(FIG_DIR / "fig06_affine_ct_to_mri_qc")
    plt.close(fig)


def plot_transducer_phase(triplet: BrainTriplet, phase: PhaseCorrection) -> None:
    fig = plt.figure(figsize=(9.5, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    pos = phase.element_positions_m * 1.0e3
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=phase.phases_rad, s=6, cmap="twilight")
    ax.scatter([0], [0], [0], c="red", s=25, marker="+")
    ax.set_title("1024-element hemispherical array")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="phase rad")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(phase.skull_lengths_m * 1.0e3, bins=32, color="#4c78a8", alpha=0.85)
    ax2.set_title("Per-element skull path length")
    ax2.set_xlabel("length mm")
    ax2.set_ylabel("elements")
    text = (
        f"target index {triplet.target_index}\n"
        f"phase span {np.ptp(phase.phases_rad):.2f} rad\n"
        f"mean skull path {np.mean(phase.skull_lengths_m)*1e3:.2f} mm\n"
        f"mean amplitude {np.mean(phase.amplitude_weights):.2f}"
    )
    ax2.text(0.98, 0.98, text, ha="right", va="top", transform=ax2.transAxes, fontsize=8)
    fig.tight_layout()
    savefig(FIG_DIR / "fig02_insightec_phase_correction")
    plt.close(fig)


def plot_essential_tremor_result(
    triplet: BrainTriplet,
    acoustic: AcousticResult,
    thermal: ThermalResult,
) -> None:
    z = triplet.target_index[2]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    panels = [
        ("pressure MPa", acoustic.pressure_pa[:, :, z] / 1.0e6, "magma"),
        ("mechanical index", acoustic.mechanical_index[:, :, z], "viridis"),
        ("peak temperature C", thermal.peak_temperature_c[:, :, z], "inferno"),
        ("CEM43 min", np.log10(thermal.cem43_min[:, :, z] + 1.0e-6), "plasma"),
    ]
    for ax, (title, data, cmap) in zip(axes.ravel(), panels):
        image = ax.imshow(np.rot90(data), cmap=cmap, origin="lower")
        ax.contour(np.rot90(thermal.lesion_mask[:, :, z]), levels=[0.5], colors=["cyan"], linewidths=0.8)
        x, y = rot90_xy((triplet.target_index[0], triplet.target_index[1]), data.shape)
        ax.plot(x, y, "w+", ms=8, mew=1.3)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.tight_layout()
    savefig(FIG_DIR / "fig03_essential_tremor_ablation")
    plt.close(fig)


def plot_gbm_plan(tumor: np.ndarray, plan: SubspotPlan) -> None:
    z = int(np.rint(np.argwhere(tumor).mean(axis=0)[2]))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(np.rot90(tumor[:, :, z]), cmap="gray", origin="lower")
    on_slice = plan.indices[plan.indices[:, 2] == z]
    if on_slice.size:
        ax.scatter(on_slice[:, 0], tumor.shape[1] - 1 - on_slice[:, 1], s=18, c="#d95f02")
    ax.set_title(f"GBM subspots, coverage {plan.covered_fraction:.2%}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    savefig(FIG_DIR / "fig04_gbm_subspot_plan")
    plt.close(fig)


def plot_gbm_bbb_opening(tumor: np.ndarray, result: BbbOpeningResult) -> None:
    z = int(np.rint(np.argwhere(tumor).mean(axis=0)[2]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    panels = [
        ("tumor mask", tumor[:, :, z].astype(float), "gray"),
        ("BBB permeability", result.permeability[:, :, z], "viridis"),
        ("inertial risk", result.inertial_cavitation_risk[:, :, z], "magma"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        image = ax.imshow(np.rot90(data), cmap=cmap, origin="lower", vmin=0.0)
        ax.contour(np.rot90(result.opened_mask[:, :, z]), levels=[0.5], colors=["cyan"], linewidths=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if title != "tumor mask":
            fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle("Segmented GBM BBB-opening subspot treatment")
    fig.tight_layout()
    savefig(FIG_DIR / "fig05_gbm_bbb_opening")
    plt.close(fig)
