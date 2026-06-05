from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
from scipy.ndimage import binary_erosion, generate_binary_structure

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
    """
    3-D phase-map of the 1024-element transcranial hemisphere and per-element
    skull path length histogram.

    The 3D panel shows element positions in physical coordinates (mm) relative
    to the VIM-like focal target (origin).  z > 0 is superior (calvarium apex);
    elements distributed at z > 0 confirm correct calvarium coverage.
    The skull surface is overlaid as a semi-transparent gray scatter.
    """
    fig = plt.figure(figsize=(12.0, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    # Skull surface for spatial context (6-connected erosion).
    skull = np.asarray(triplet.skull_mask, dtype=bool)
    struct6 = generate_binary_structure(3, 1)
    skull_surface = skull & ~binary_erosion(skull, structure=struct6, iterations=1)
    surf_idx = np.argwhere(skull_surface)
    if len(surf_idx) > 4000:
        surf_idx = surf_idx[::max(1, len(surf_idx) // 4000)]
    sp = np.asarray(triplet.ct_hu.spacing_m)   # (sx, sy, sz) in m
    nx, ny, nz = skull.shape
    # Convert voxel index to physical mm centred on volume
    surf_mm = (surf_idx - np.array([(nx - 1) * 0.5, (ny - 1) * 0.5, (nz - 1) * 0.5])) * sp * 1e3
    ax.scatter(surf_mm[:, 0], surf_mm[:, 1], surf_mm[:, 2],
               c="silver", s=0.6, alpha=0.12, rasterized=True, label="Skull surface")

    # Element positions colored by phase.
    pos_mm = phase.element_positions_m * 1.0e3
    sc = ax.scatter(pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2],
                    c=phase.phases_rad, s=8, cmap="twilight", zorder=3,
                    label="Array elements (phase)")
    ax.scatter([0], [0], [0], c="lime", s=50, marker="*", zorder=5, label="VIM target")

    # Superior (z > 0) annotation confirms calvarium coverage.
    z_mean_mm = float(np.mean(pos_mm[:, 2]))
    ax.set_title(
        f"{phase.element_positions_m.shape[0]}-element transcranial cap\n"
        f"(mean z = {z_mean_mm:+.1f} mm; z > 0 = calvarium/superior)"
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm] superior")  # type: ignore[attr-defined]
    ax.legend(fontsize=7, markerscale=2, loc="upper left")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="phase [rad]")

    # Skull path length histogram.
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(phase.skull_lengths_m * 1.0e3, bins=32, color="#4c78a8", alpha=0.85)
    ax2.set_title("Per-element skull path length")
    ax2.set_xlabel("length [mm]")
    ax2.set_ylabel("elements")
    text = (
        f"target index {triplet.target_index}\n"
        f"phase span {np.ptp(phase.phases_rad):.2f} rad\n"
        f"mean skull path {np.mean(phase.skull_lengths_m)*1e3:.2f} mm\n"
        f"mean amplitude {np.mean(phase.amplitude_weights):.2f}"
    )
    ax2.text(0.98, 0.98, text, ha="right", va="top", transform=ax2.transAxes, fontsize=8)
    fig.tight_layout()
    savefig(FIG_DIR / "fig02_transcranial_bowl_phase_correction")
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


def plot_focused_bowl_calvarium_3d(triplet: BrainTriplet, phase: PhaseCorrection) -> None:
    """
    3-D visualization of the 1024-element hemispherical focused-bowl cap
    covering the calvarium.

    Physical model
    --------------
    The source layout is a hemispherical focused-bowl cap centered on the
    acoustic focus. Element positions are at z > 0 (superior,
    calvarium/top-of-skull) with polar angles from cap_min near the apex to
    cap_max near the lateral rim.

    Coordinate convention:  z > 0 superior (apex of calvarium),
    z < 0 inferior (base of skull / neck). The source cap sits above z = 0
    and covers the calvarium.

    The figure overlays:
    - White semi-transparent scatter: outer skull surface voxels (calvarium).
    - Red scatter: focused-bowl elements (z > 0, on calvarium).
    - Green star: acoustic focus (VIM-like target inside brain).
    - Dashed blue line: beam axis from skull vertex to focus.

    Invariant verified:  all element z-coordinates are positive (superior).

    The source-domain model remains a generic focused-bowl cap; anatomy enters
    only through the placement target and CT-derived skull support.
    """
    pos_mm = phase.element_positions_m * 1.0e3  # (N, 3) mm

    # Skull surface extraction (6-connected boundary).
    # Both skull surface and element positions must share the same reference
    # frame.  Element positions from fibonacci_hemisphere / phase_correction_through_ct
    # are expressed relative to the target (focus) index via index_to_point:
    #   point_m = (voxel_index - target_index) * spacing_m
    # We use the same convention here so both point clouds share origin = focus.
    skull = np.asarray(triplet.skull_mask, dtype=bool)
    struct6 = generate_binary_structure(3, 1)
    skull_surface = skull & ~binary_erosion(skull, structure=struct6, iterations=1)
    surf_idx = np.argwhere(skull_surface)
    if len(surf_idx) > 5000:
        surf_idx = surf_idx[::max(1, len(surf_idx) // 5000)]
    sp = np.asarray(triplet.ct_hu.spacing_m)          # [sx, sy, sz] in m
    ti = np.asarray(triplet.target_index, dtype=float)  # focus voxel index
    # physical mm relative to focus (same frame as element_positions_m * 1e3)
    surf_mm = (surf_idx.astype(float) - ti) * sp * 1e3  # (M, 3) mm

    # Calvarium surface: restrict to z > 0 (superior half) for clarity
    superior_mask = surf_mm[:, 2] > 0.0
    surf_cal = surf_mm[superior_mask]

    # Invariant: verify all elements are at z > 0.
    n_inferior = int(np.sum(pos_mm[:, 2] <= 0.0))
    assert n_inferior == 0, (
        f"fibonacci_hemisphere orientation error: {n_inferior} elements at z <= 0 "
        "(inferior/neck side). All elements must be at z > 0 (calvarium)."
    )

    # VIM focus: the origin in element-position space.
    # focus_mm = (target_index - target_index) * sp * 1e3 = [0, 0, 0]
    focus_mm = np.zeros(3, dtype=np.float64)

    # 3D figure.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Skull calvarium surface: semi-transparent white.
    if len(surf_cal) > 0:
        ax.scatter(surf_cal[:, 0], surf_cal[:, 1], surf_cal[:, 2],
                   c="whitesmoke", s=1.5, alpha=0.15, rasterized=True,
                   label="Calvarium surface")

    # Transducer elements: red, all on superior (calvarium) side.
    ax.scatter(pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2],
               c="#d62728", s=6, alpha=0.85, zorder=4,
               label=f"Array elements ({len(pos_mm)}, calvarium)")

    # VIM acoustic focus
    ax.scatter(*focus_mm, c="lime", s=100, marker="*", zorder=6,
               label="VIM-like focus (intracranial)")

    # Bowl apex (point nearest to superior pole = max z element)
    apex_idx = int(np.argmax(pos_mm[:, 2]))
    apex_mm = pos_mm[apex_idx]
    ax.scatter(*apex_mm, c="orange", s=80, marker="^", zorder=6,
               label="Bowl apex (skull contact)")
    ax.plot([apex_mm[0], focus_mm[0]],
            [apex_mm[1], focus_mm[1]],
            [apex_mm[2], focus_mm[2]],
            "b--", lw=0.9, alpha=0.55, label="Beam axis")

    radius_mm = float(np.linalg.norm(pos_mm[0]))
    z_min_mm = float(pos_mm[:, 2].min())
    z_max_mm = float(pos_mm[:, 2].max())
    ax.set_title(
        f"{len(pos_mm)}-element hemispherical focused-bowl cap\n"
        f"covering calvarium; z in [{z_min_mm:+.0f}, {z_max_mm:+.0f}] mm "
        f"(z > 0 = superior); R = {radius_mm:.0f} mm"
    )
    ax.set_xlabel("x [mm] (left-right)")
    ax.set_ylabel("y [mm] (anterior-posterior)")
    ax.set_zlabel("z [mm] superior")  # type: ignore[attr-defined]
    ax.legend(fontsize=8, markerscale=2, loc="lower left")

    # View from oblique-superior: look down and slightly anterior so both the
    # focused-bowl cap and the brain beneath are visible.
    ax.view_init(elev=25, azim=-70)

    # Equal-aspect bounding box
    all_pts = np.vstack([pos_mm, surf_cal[:200] if len(surf_cal) >= 200 else surf_cal])
    cx, cy, cz = float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1])), float(np.mean(all_pts[:, 2]))
    r_eq = float(np.abs(all_pts - np.array([cx, cy, cz])).max()) * 1.08
    ax.set_xlim(cx - r_eq, cx + r_eq)
    ax.set_ylim(cy - r_eq, cy + r_eq)
    ax.set_zlim(cz - r_eq, cz + r_eq)  # type: ignore[attr-defined]

    fig.tight_layout()
    savefig(FIG_DIR / "fig00_focused_bowl_calvarium_3d")
    plt.close(fig)
