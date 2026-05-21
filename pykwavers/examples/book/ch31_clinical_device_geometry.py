"""Chapter 31: Clinical Theranostic Device Geometries — 3-D Transducer-Patient Integration.

All geometry is computed by kwavers (Rust) through the PyO3 wrappers:
  - ``plan_abdominal_array_placement_from_ritk_ct`` for liver and kidney.
  - ``plan_transcranial_focused_bowl_placement_from_ritk_ct`` for the transcranial focused bowl.
  - ``run_theranostic_inverse_from_ritk`` for simulated exposures and reconstructions.

Python owns only figure rendering and file I/O.  No physics computation is in this file.

Device naming note: the "HistoSonics-like" label refers to the bowl geometry
(focused hemispherical array, central imaging cutout, anterior skin placement)
common to research histotripsy systems.  The "InSightec-like" label refers to the
transcranial phased-array focused-bowl geometry (1024 elements, calvarium coverage,
CT-planned skull-entry correction) common to Exablate Neuro platforms.  Neither
label implies proprietary device specifications.

Coordinate convention: all physical coordinates are in metres with origin at the
body centroid of each CT volume.  Positive x is patient right, positive y is
anterior, positive z is superior (standard LPS orientation).

Figures produced:
  fig01_liver_array_3d_geometry.png  — HistoSonics-like bowl on liver CT
  fig02_kidney_array_3d_geometry.png — HistoSonics-like bowl on kidney CT
  fig03_brain_focused_bowl_3d_calvarium.png — InSightec-like focused bowl at calvarium level
  fig04_exposure_comparison.png      — simulated pressure for all three anatomies
  fig05_reconstruction_metrics.png   — reconstruction fidelity metrics
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
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch31"
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
if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))

import pykwavers as kw  # noqa: E402
from transcranial_planning.scene import CANONICAL_BRAIN_SCENE  # noqa: E402


# ── Data paths ─────────────────────────────────────────────────────────────────

LIVER_CT = REPO_ROOT / "data" / "lits17_sample" / "volume-0.nii"
LIVER_SEG = REPO_ROOT / "data" / "lits17_sample" / "segmentation-0.nii"

KIDNEY_CT = REPO_ROOT / "data" / "kits19_sample" / "case_00000.nii.gz"
KIDNEY_SEG = REPO_ROOT / "data" / "kits19_sample" / "segmentation_00000.nii.gz"

BRAIN_CT = REPO_ROOT / "data" / "rire_patient_109" / "patient_109_ct.nii.gz"


# ── Chapter-31 specific parameters ─────────────────────────────────────────────

ABDOMEN_ELEMENTS = int(os.environ.get("KWAVERS_CH31_ABDOMEN_ELEMENTS", "256"))
BRAIN_ELEMENTS = CANONICAL_BRAIN_SCENE.transducer.element_count
SURFACE_STRIDE = int(os.environ.get("KWAVERS_CH31_SURFACE_STRIDE", "8"))
BRAIN_SURFACE_STRIDE = int(os.environ.get("KWAVERS_CH31_BRAIN_SURFACE_STRIDE", "6"))
BODY_HU_THRESHOLD = float(os.environ.get("KWAVERS_CH31_BODY_HU", "-400.0"))

# Exposure/reconstruction parameters (small defaults for quick runs).
ABDOMEN_GRID = int(os.environ.get("KWAVERS_CH31_ABDOMEN_GRID", "52"))
BRAIN_GRID = int(os.environ.get("KWAVERS_CH31_BRAIN_GRID", "48"))
ITERATIONS = int(os.environ.get("KWAVERS_CH31_ITERATIONS", "10"))


# ── Geometry planning ───────────────────────────────────────────────────────────

def plan_liver_geometry() -> dict[str, object]:
    return kw.plan_abdominal_array_placement_from_ritk_ct(
        str(LIVER_CT),
        str(LIVER_SEG),
        anatomy_label="liver",
        element_count=ABDOMEN_ELEMENTS,
        surface_stride=SURFACE_STRIDE,
        body_hu_threshold=BODY_HU_THRESHOLD,
    )


def plan_kidney_geometry() -> dict[str, object]:
    return kw.plan_abdominal_array_placement_from_ritk_ct(
        str(KIDNEY_CT),
        str(KIDNEY_SEG),
        anatomy_label="kidney",
        element_count=ABDOMEN_ELEMENTS,
        surface_stride=SURFACE_STRIDE,
        body_hu_threshold=BODY_HU_THRESHOLD,
    )


def plan_brain_geometry() -> dict[str, object]:
    return kw.plan_transcranial_focused_bowl_placement_from_ritk_ct(
        str(BRAIN_CT),
        surface_stride=BRAIN_SURFACE_STRIDE,
        **CANONICAL_BRAIN_SCENE.focused_bowl_pykwavers_kwargs(),
    )


# ── Exposure and reconstruction ────────────────────────────────────────────────

def run_abdomen_case(ct_path: Path, seg_path: Path, anatomy: str) -> dict[str, object]:
    return kw.run_theranostic_inverse_from_ritk(
        str(ct_path),
        str(seg_path),
        anatomy=anatomy,
        grid_size=ABDOMEN_GRID,
        element_count=ABDOMEN_ELEMENTS,
        iterations=ITERATIONS,
        frequencies_hz=[250_000.0, 500_000.0, 750_000.0],
        receiver_offsets=[32, 64, 96, 128],
        source_pressure_pa=28.0e6,
        noise_fraction=float(os.environ.get("KWAVERS_CH31_NOISE_FRACTION", "0.012")),
        inverse_encoding_rows_per_code=2,
    )


def run_brain_case() -> dict[str, object]:
    return kw.run_theranostic_inverse_from_ritk(
        str(BRAIN_CT),
        None,
        anatomy="brain",
        grid_size=BRAIN_GRID,
        element_count=BRAIN_ELEMENTS,
        iterations=ITERATIONS,
        frequencies_hz=[220_000.0, CANONICAL_BRAIN_SCENE.transducer.frequency_hz],
        receiver_offsets=[256, 384, 512, 640],
        source_pressure_pa=CANONICAL_BRAIN_SCENE.transducer.diagnostic_source_pressure_pa,
        noise_fraction=float(os.environ.get("KWAVERS_CH31_NOISE_FRACTION", "0.012")),
        inverse_encoding_rows_per_code=2,
        target_fraction_xyz=CANONICAL_BRAIN_SCENE.target.fraction_xyz,
    )


# ── Water coupling medium helper ───────────────────────────────────────────────

def _draw_water_coupling(
    ax: "plt.Axes",
    focus: np.ndarray,
    skin: np.ndarray,
    bowl_radius_m: float,
    n_theta: int = 30,
    n_phi: int = 60,
) -> None:
    """Render the water coupling medium as a semi-transparent spherical cap.

    The water fills the interior of the bowl from the skin surface to the
    transducer face.  The cap surface is parameterised by:

        P(θ,φ) = F − R·[cos(θ)·d̂ + sin(θ)·(cos(φ)·ê₁ + sin(φ)·ê₂)]

    where d̂ = (F − S)/‖F − S‖ is the inward bowl axis (skin → focus),
    θ ∈ [0, θ_max] covers from vertex to rim, and (ê₁, ê₂) is a Gram–Schmidt
    frame perpendicular to d̂.
    """
    THETA_MAX = 0.960  # ≈ 55°, matches Rust constant

    focal_depth = float(np.linalg.norm(focus - skin))
    if focal_depth < 1e-6:
        return

    # d̂: inward axis from skin toward focus
    d_hat = (focus - skin) / focal_depth

    # Gram–Schmidt orthonormal frame (ê₁, ê₂) ⊥ d̂
    arb = np.array([0.0, 0.0, 1.0]) if abs(d_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d_hat, arb)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d_hat, e1)

    # Meshgrid over (θ, φ)
    theta = np.linspace(0.0, THETA_MAX, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)

    cos_th = np.cos(TH)
    sin_th = np.sin(TH)
    cos_ph = np.cos(PH)
    sin_ph = np.sin(PH)

    R = bowl_radius_m
    X = focus[0] - R * (cos_th * d_hat[0] + sin_th * (cos_ph * e1[0] + sin_ph * e2[0]))
    Y = focus[1] - R * (cos_th * d_hat[1] + sin_th * (cos_ph * e1[1] + sin_ph * e2[1]))
    Z = focus[2] - R * (cos_th * d_hat[2] + sin_th * (cos_ph * e1[2] + sin_ph * e2[2]))

    ax.plot_surface(
        X, Y, Z,
        color="#29b6f6",  # water blue
        alpha=0.14,
        linewidth=0,
        antialiased=True,
        shade=True,
        zorder=1,
    )


# ── 3-D abdominal bowl rendering ────────────────────────────────────────────────

def render_abdominal_3d(
    geo: dict[str, object],
    anatomy_label: str,
    fig_path: Path,
) -> Path:
    """3-D scatter visualisation showing the bowl transducer on the skin surface.

    The body skin surface and organ surface are shown as point clouds.  The bowl
    elements sit outside the body, visible on the skin.  Beam lines connect
    elements to the organ centroid (focus).  The skin contact point (bowl vertex)
    is marked with a circle.
    """
    body = np.asarray(geo["body_surface_points_m"], dtype=float)
    organ = np.asarray(geo["organ_surface_points_m"], dtype=float)
    elements = np.asarray(geo["therapy_elements_m"], dtype=float)
    starts = np.asarray(geo["beam_start_points_m"], dtype=float)
    ends = np.asarray(geo["beam_end_points_m"], dtype=float)
    focus = np.asarray(geo["focus_m"], dtype=float)
    skin = np.asarray(geo["skin_contact_m"], dtype=float)

    # Focal depth = skin-to-focus distance [mm]; bowl radius = R = focal_depth/cos(θ_max).
    focal_depth_mm = 1e3 * float(np.linalg.norm(focus - skin))
    bowl_radius_mm = 1e3 * float(geo["transducer_radius_m"])

    fig = plt.figure(figsize=(14.0, 6.0), constrained_layout=True)
    fig.suptitle(
        f"HistoSonics-like bowl on {anatomy_label} CT — {int(geo['element_count'])} elements, "
        f"focal depth {focal_depth_mm:.0f} mm, bowl radius {bowl_radius_mm:.0f} mm",
        fontsize=11,
    )

    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2, projection="3d")

    # Camera positioning — place camera on the SAME side as the bowl so the bowl
    # elements appear in front of the body (not occluded behind it).
    #
    # n̂ = (skin − focus) / ‖skin − focus‖ is the outward bowl axis (from focus
    # toward skin contact, i.e. pointing out of the body at the bowl face).
    # Matplotlib's azimuth places the camera at angle `azim` in the x-y plane,
    # looking TOWARD the origin.  Setting azim = atan2(n̂_y, n̂_x) positions the
    # camera in the same angular direction as the bowl; elements between camera
    # and body are visible in front of the body cloud.
    bowl_axis = skin - focus  # outward from body toward bowl
    bowl_axis_norm = bowl_axis / (np.linalg.norm(bowl_axis) + 1e-12)

    # Primary azimuth: camera on the bowl side (NO +180 offset).
    # Without +180 the camera is in the same angular direction as n̂, so elements
    # between camera and body appear in front of the body cloud.
    azim_primary = float(np.degrees(np.arctan2(bowl_axis_norm[1], bowl_axis_norm[0])))
    # Elevation scales with the vertical component of the bowl axis:
    #   |n̂_z| ≈ 1 (superior bowl) → high elev (~65°) for top-down view of the dome
    #   |n̂_z| ≈ 0 (lateral bowl) → low elev (~15°) for horizontal profile of the cap
    bowl_z = float(np.clip(bowl_axis_norm[2], -1.0, 1.0))
    elev_primary = float(np.clip(15.0 + 55.0 * abs(bowl_z), 12.0, 70.0))

    # Side azimuth: 90° from bowl horizontal so we see the bowl in profile.
    bowl_horiz = np.array([bowl_axis_norm[0], bowl_axis_norm[1], 0.0])
    horiz_len = float(np.linalg.norm(bowl_horiz))
    if horiz_len > 0.15:
        bowl_horiz /= horiz_len
        azim_side = float(np.degrees(np.arctan2(bowl_horiz[1], bowl_horiz[0]))) + 90.0
    else:
        # Bowl is nearly vertical — any horizontal angle gives a profile view.
        azim_side = azim_primary + 90.0

    for ax in (ax_a, ax_b):
        # Water coupling medium — semi-transparent spherical cap (bowl interior).
        # Drawn first so it appears behind the body and elements.
        _draw_water_coupling(ax, focus, skin, float(geo["transducer_radius_m"]))

        # Body skin surface — very transparent so bowl elements in front are visible.
        if body.size:
            ax.scatter(
                body[:, 0], body[:, 1], body[:, 2],
                s=0.4, c="#8090a0", alpha=0.07, depthshade=False, rasterized=True,
            )
        # Organ surface — amber.
        if organ.size:
            ax.scatter(
                organ[:, 0], organ[:, 1], organ[:, 2],
                s=3.0, c="#f5a623", alpha=0.55, depthshade=False,
                label=f"{anatomy_label} surface",
            )
        # Bowl elements — red, outside the body (larger markers for visibility).
        if elements.size:
            ax.scatter(
                elements[:, 0], elements[:, 1], elements[:, 2],
                s=8.0, c="#d0021b", alpha=0.85, depthshade=False,
                label=f"therapy elements ({int(geo['element_count'])})",
                zorder=4,
            )
        # Water coupling medium label — proxy Line2D (add_patch not supported on Axes3D).
        import matplotlib.lines as mlines  # noqa: PLC0415
        ax._water_proxy = mlines.Line2D(  # stored so legend can reference it
            [], [], marker="s", color="w", markerfacecolor="#29b6f6",
            markeredgecolor="#1e88e5", markersize=8, alpha=0.8,
            label="water coupling medium",
        )

        # Skin contact point — lime circle (bowl vertex on skin).
        ax.scatter(
            [skin[0]], [skin[1]], [skin[2]],
            marker="o", s=120, c="lime", edgecolors="black", linewidths=0.8,
            zorder=6, label="skin contact (bowl rim centre)",
        )
        # Focus marker — cyan cross.
        ax.scatter(
            [focus[0]], [focus[1]], [focus[2]],
            marker="x", s=120, c="cyan", linewidths=2.5, zorder=6,
            label="focus (organ centroid)",
        )
        # Beam lines — semi-transparent orange.
        for start, end in zip(starts, ends):
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                c="#ff6b00", lw=0.4, alpha=0.25,
            )
        _set_equal_3d_limits(ax, [body, organ, elements])
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)

    # Oblique view: camera on the bowl side, elevated for depth.
    ax_a.view_init(elev=elev_primary, azim=azim_primary)
    ax_a.set_title(
        "Oblique view — bowl on skin surface\n"
        "(red elements outside body, lime = skin contact, cyan = organ focus)",
        fontsize=9,
    )
    handles, labels = ax_a.get_legend_handles_labels()
    handles.insert(0, ax_a._water_proxy)
    labels.insert(0, ax_a._water_proxy.get_label())
    ax_a.legend(handles, labels, loc="lower left", fontsize=6, frameon=True)

    # Side view: 90° from bowl axis to show profile (bowl visible on body edge).
    ax_b.view_init(elev=15, azim=azim_side)
    ax_b.set_title(
        "Side view — transducer profile\n"
        "bowl sits outside skin; beams converge at organ centroid",
        fontsize=9,
    )

    _save_figure(fig, fig_path)
    plt.close(fig)
    return fig_path


# ── Brain focused-bowl rendering (calvarium-level) ──────────────────────────────

def render_brain_focused_bowl_3d(geo: dict[str, object], fig_path: Path) -> Path:
    """Render the 1024-element InSightec-like focused bowl covering the calvarium.

    The view is tilted to show the elements ring the upper skull (calvarium), not
    the neck.  Skull-beam intersection points (yellow) confirm the beams enter the
    calvarium rather than the base.  Two views: (left) slightly elevated to show
    the full calvarium shell; (right) directly above to confirm circular aperture.
    """
    head = np.asarray(geo["head_surface_points_m"], dtype=float)
    skull = np.asarray(geo["skull_surface_points_m"], dtype=float)
    elements = np.asarray(geo["therapy_elements_m"], dtype=float)
    starts = np.asarray(geo["beam_start_points_m"], dtype=float)
    ends = np.asarray(geo["beam_end_points_m"], dtype=float)
    intersections = np.asarray(geo["skull_intersections_m"], dtype=float)
    focus = np.asarray(geo["focus_m"], dtype=float)

    fig = plt.figure(figsize=(14.0, 6.0), constrained_layout=True)
    fig.suptitle(
        f"InSightec-like focused bowl at calvarium level — {int(geo['element_count'])} elements, "
        f"bowl radius {1e3 * float(geo['bowl_radius_m']):.0f} mm, "
        f"skull entry fraction {float(geo['intersection_fraction']):.2f}",
        fontsize=11,
    )

    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2, projection="3d")

    clouds = [c for c in [head, skull, elements] if c.size]

    for ax in (ax_a, ax_b):
        if head.size:
            ax.scatter(
                head[:, 0], head[:, 1], head[:, 2],
                s=0.8, c="#b8c0cc", alpha=0.12, depthshade=False, rasterized=True,
            )
        if skull.size:
            ax.scatter(
                skull[:, 0], skull[:, 1], skull[:, 2],
                s=1.2, c="#f2d7a0", alpha=0.35, depthshade=False, label="skull surface",
            )
        if elements.size:
            ax.scatter(
                elements[:, 0], elements[:, 1], elements[:, 2],
                s=5.0, c="#d94f45", alpha=0.65, depthshade=False, label="bowl elements",
            )
        if intersections.size:
            ax.scatter(
                intersections[:, 0], intersections[:, 1], intersections[:, 2],
                s=20, c="#ffff33", edgecolors="black", linewidths=0.25, depthshade=False,
                label="skull entry points",
            )
        ax.scatter(
            [focus[0]], [focus[1]], [focus[2]],
            marker="x", s=90, c="white", linewidths=2.2, zorder=5, label="focus",
        )
        for start, end in zip(starts, ends):
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                c="#ff8c00", lw=0.5, alpha=0.40,
            )
        _set_equal_3d_limits(ax, clouds)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)

    # Elevated oblique: clearly shows elements wrap over the top of the head.
    ax_a.view_init(elev=35, azim=-60)
    ax_a.set_title(
        "Oblique view — bowl covers calvarium\n"
        "(elements encircle top of skull, not neck)",
        fontsize=9,
    )
    ax_a.legend(loc="lower left", fontsize=6, frameon=True)

    # Top-down: confirms the aperture is a complete cap over the vertex.
    ax_b.view_init(elev=78, azim=0)
    ax_b.set_title(
        "Top-down view — calvarium aperture\n"
        "circular element distribution around vertex",
        fontsize=9,
    )

    _save_figure(fig, fig_path)
    plt.close(fig)
    return fig_path


# ── Exposure comparison ─────────────────────────────────────────────────────────

def render_exposure_comparison(
    liver_result: dict[str, object],
    kidney_result: dict[str, object],
    brain_result: dict[str, object],
    fig_path: Path,
) -> Path:
    """Side-by-side simulated exposure maps for all three anatomies."""
    cases = [
        (liver_result, "liver (HistoSonics-like)", "#f5a623"),
        (kidney_result, "kidney (HistoSonics-like)", "#7b68ee"),
        (brain_result, "brain (InSightec-like)", "#4a9edd"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5), constrained_layout=True)
    fig.suptitle("Simulated pressure exposure — CT-derived heterogeneous media", fontsize=12)

    for ax, (result, label, _color) in zip(axes, cases):
        exposure = np.asarray(result["exposure"], dtype=float)
        ct = np.asarray(result["placement_ct_hu"], dtype=float)
        spacing = tuple(float(v) for v in result["placement_spacing_m"])
        extent = _image_extent_xy(ct, spacing)

        ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300, alpha=0.55)
        im = ax.imshow(
            np.abs(exposure).T, cmap="inferno", origin="lower", extent=extent,
            vmin=0.0, vmax=float(np.max(np.abs(exposure))) or 1.0, alpha=0.80,
        )
        # Target contour.
        target = np.asarray(result.get("target_mask", np.zeros_like(ct)), dtype=bool)
        if target.any():
            x = np.linspace(extent[0], extent[1], target.shape[0])
            y = np.linspace(extent[2], extent[3], target.shape[1])
            ax.contour(x, y, target.T.astype(float), levels=[0.5], colors="white", linewidths=0.9)

        # Therapy element positions (2D projection).
        for key, color, size in (
            ("placement_therapy_points_m", "#e74c3c", 1.5),
            ("placement_imaging_points_m", "#2e86de", 5.0),
        ):
            pts = np.asarray(result.get(key, []), dtype=float)
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, alpha=0.55)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="pressure [Pa]")

    _save_figure(fig, fig_path)
    plt.close(fig)
    return fig_path


# ── Reconstruction metrics ─────────────────────────────────────────────────────

def render_reconstruction_metrics(
    liver_result: dict[str, object],
    kidney_result: dict[str, object],
    brain_result: dict[str, object],
    fig_path: Path,
) -> Path:
    """Bar chart of Dice (equal-area threshold) and CNR for each anatomy/channel."""
    channel_keys = [
        ("active_lesion", "active Born"),
        ("waveform_rtm", "linear RTM"),
        ("subharmonic", "subharmonic"),
        ("harmonic", "harmonic"),
        ("ultraharmonic", "ultraharmonic"),
        ("fusion", "fusion"),
    ]
    cases = [
        (liver_result, "liver"),
        (kidney_result, "kidney"),
        (brain_result, "brain"),
    ]

    n_channels = len(channel_keys)
    n_cases = len(cases)
    x = np.arange(n_channels)
    width = 0.25
    colors = ["#f5a623", "#7b68ee", "#4a9edd"]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0), constrained_layout=True)
    fig.suptitle("Reconstruction fidelity — Dice (equal-area) and CNR", fontsize=12)

    for metric_idx, (ax, metric_label, metric_key) in enumerate(
        [
            (axes[0], "Dice (equal-area threshold)", "dice_equal_area"),
            (axes[1], "CNR", "cnr"),
        ]
    ):
        for case_idx, (result, label) in enumerate(cases):
            values = []
            for ch_key, _ in channel_keys:
                m = result.get("metrics", {}).get(ch_key, {})
                if not m:
                    m = result.get("metrics", {}).get("fusion", {})
                values.append(float(m.get(metric_key, 0.0)))
            bars = ax.bar(x + case_idx * width, values, width, label=label, color=colors[case_idx], alpha=0.82)
            ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)

        ax.set_xlabel("Reconstruction channel", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(metric_label, fontsize=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels([ch_name for _, ch_name in channel_keys], rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        if metric_idx == 0:
            ax.set_ylim(0.0, 1.05)

    _save_figure(fig, fig_path)
    plt.close(fig)
    return fig_path


# ── Metrics JSON ───────────────────────────────────────────────────────────────

def write_metrics(
    liver_geo: dict[str, object],
    kidney_geo: dict[str, object],
    brain_geo: dict[str, object],
    liver_result: dict[str, object],
    kidney_result: dict[str, object],
    brain_result: dict[str, object],
    figures: list[Path],
) -> Path:
    liver_elements = np.asarray(liver_geo["therapy_elements_m"], dtype=float)
    kidney_elements = np.asarray(kidney_geo["therapy_elements_m"], dtype=float)
    brain_elements = np.asarray(brain_geo["therapy_elements_m"], dtype=float)

    payload: dict[str, object] = {
        "chapter": 31,
        "analysis": (
            "3-D clinical device geometry: HistoSonics-like bowl on liver/kidney skin, "
            "InSightec-like focused bowl at calvarium; fully simulated exposures and reconstructions "
            "via kwavers PyO3 API with RITK NIfTI/DICOM ingestion"
        ),
        "figures": [str(p) for p in figures],
        "brain_scene": CANONICAL_BRAIN_SCENE.to_manifest(),
        "geometries": {
            "liver": {
                "anatomy_label": str(liver_geo["anatomy_label"]),
                "geometry_model": str(liver_geo["geometry_model"]),
                "element_count": int(liver_geo["element_count"]),
                "transducer_radius_m": float(liver_geo["transducer_radius_m"]),
                "body_surface_points": int(liver_elements.shape[0] if liver_elements.ndim == 2 else 0),
                "skin_contact_m": [float(v) for v in liver_geo["skin_contact_m"]],
                "focus_m": [float(v) for v in liver_geo["focus_m"]],
            },
            "kidney": {
                "anatomy_label": str(kidney_geo["anatomy_label"]),
                "geometry_model": str(kidney_geo["geometry_model"]),
                "element_count": int(kidney_geo["element_count"]),
                "transducer_radius_m": float(kidney_geo["transducer_radius_m"]),
                "body_surface_points": int(kidney_elements.shape[0] if kidney_elements.ndim == 2 else 0),
                "skin_contact_m": [float(v) for v in kidney_geo["skin_contact_m"]],
                "focus_m": [float(v) for v in kidney_geo["focus_m"]],
            },
            "brain": {
                "geometry_model": str(brain_geo["geometry_model"]),
                "element_count": int(brain_geo["element_count"]),
                "bowl_radius_m": float(brain_geo["bowl_radius_m"]),
                "skull_intersection_fraction": float(brain_geo["intersection_fraction"]),
                "skull_hu_threshold": float(brain_geo["skull_hu_threshold"]),
                "target_fraction_xyz": [float(v) for v in brain_geo.get("target_fraction_xyz", ())],
                "body_surface_points": int(brain_elements.shape[0] if brain_elements.ndim == 2 else 0),
            },
        },
        "inverse_results": {
            "liver": _summarise_inverse(liver_result),
            "kidney": _summarise_inverse(kidney_result),
            "brain": _summarise_inverse(brain_result),
        },
    }
    path = OUT_DIR / "metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _summarise_inverse(result: dict[str, object]) -> dict[str, object]:
    return {
        "device_model": str(result.get("device_model", "")),
        "element_count": int(result.get("element_count", 0)),
        "source_pressure_pa": float(result.get("source_pressure_pa", 0.0)),
        "target_fraction_xyz": [float(v) for v in result.get("target_fraction_xyz", ())],
        "active_voxels": int(result.get("active_voxels", 0)),
        "measurements": int(result.get("measurements", 0)),
        "metrics": result.get("metrics", {}),
    }


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260)
    try:
        fig.savefig(path.with_suffix(".pdf"))
    except PermissionError:
        pass


def _image_extent_xy(image: np.ndarray, spacing_m: tuple[float, float]) -> list[float]:
    nx, ny = image.shape
    return [
        -0.5 * (nx - 1) * spacing_m[0],
        0.5 * (nx - 1) * spacing_m[0],
        -0.5 * (ny - 1) * spacing_m[1],
        0.5 * (ny - 1) * spacing_m[1],
    ]


def _set_equal_3d_limits(ax: plt.Axes, clouds: list[np.ndarray]) -> None:
    non_empty = [c for c in clouds if c.size >= 3]
    if not non_empty:
        return
    stacked = np.vstack(non_empty)
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    centre = 0.5 * (mins + maxs)
    radius = 0.54 * float(np.max(maxs - mins))
    radius = max(radius, 0.01)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Plan 3-D device geometries.
    liver_geo = plan_liver_geometry()
    kidney_geo = plan_kidney_geometry()
    brain_geo = plan_brain_geometry()

    # 2. Run simulated exposures and reconstructions.
    liver_result = run_abdomen_case(LIVER_CT, LIVER_SEG, "liver")
    kidney_result = run_abdomen_case(KIDNEY_CT, KIDNEY_SEG, "kidney")
    brain_result = run_brain_case()

    # 3. Render figures.
    figures = [
        render_abdominal_3d(liver_geo, "liver", OUT_DIR / "fig01_liver_array_3d_geometry.png"),
        render_abdominal_3d(kidney_geo, "kidney", OUT_DIR / "fig02_kidney_array_3d_geometry.png"),
        render_brain_focused_bowl_3d(brain_geo, OUT_DIR / "fig03_brain_focused_bowl_3d_calvarium.png"),
        render_exposure_comparison(
            liver_result, kidney_result, brain_result,
            OUT_DIR / "fig04_exposure_comparison.png",
        ),
        render_reconstruction_metrics(
            liver_result, kidney_result, brain_result,
            OUT_DIR / "fig05_reconstruction_metrics.png",
        ),
    ]

    metrics = write_metrics(
        liver_geo, kidney_geo, brain_geo,
        liver_result, kidney_result, brain_result,
        figures,
    )
    return {"figures": [str(p) for p in figures], "metrics": str(metrics)}


if __name__ == "__main__" or __name__ == "ch31":
    run()
