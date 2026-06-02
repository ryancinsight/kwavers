"""Chapter 31: Clinical Theranostic Device Geometries — 3-D Transducer-Patient Integration.

All geometry is computed by kwavers (Rust) through the PyO3 wrappers:
  - ``plan_abdominal_array_placement_from_ritk_ct`` for liver and kidney.
  - ``plan_transcranial_focused_bowl_placement_from_ritk_ct`` for the transcranial focused bowl.
  - ``run_theranostic_inverse_from_ritk`` for simulated exposures and reconstructions.

Python owns only figure rendering and file I/O.  No physics computation is in this file.

Device naming note: source geometry is expressed through generic focused-bowl
parameters. The abdominal cases use a skin-coupled hemispherical bowl with a
central imaging cutout; the brain case uses a 1024-element transcranial
calvarium focused-bowl cap with CT-planned skull-entry correction.

Coordinate convention: all physical coordinates are in metres with origin at the
body centroid of each CT volume.  Positive x is patient right, positive y is
anterior, positive z is superior (standard LPS orientation).

Figures produced:
  fig01_liver_array_3d_geometry.png  — skin-coupled hemispherical focused bowl on liver CT
  fig02_kidney_array_3d_geometry.png — skin-coupled hemispherical focused bowl on kidney CT
  fig03_brain_focused_bowl_3d_calvarium.png — transcranial focused bowl at calvarium level
  fig04_exposure_comparison.png      — simulated pressure for all three anatomies
  fig05_reconstruction_metrics.png   — reconstruction fidelity metrics
  fig06_liver_image_then_treat.png   — liver image-then-treat sequence (recon + focused lesion)
  fig07_kidney_image_then_treat.png  — kidney image-then-treat sequence
  fig08_brain_image_then_treat.png   — brain image-then-treat sequence
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
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

# Passive cavitation channels (subharmonic f0/2, ultraharmonic 3f0/2) use genuine
# passive acoustic mapping by default: the cavitation emission is simulated through
# the heterogeneous medium and the receiver traces are DMAS-beamformed (kwavers
# `PassiveReconstructionMode::PassiveAcousticMapping`). Set to "operator" for the
# legacy finite-frequency operator inverse.
PASSIVE_RECON = os.environ.get("KWAVERS_CH31_PASSIVE_RECON", "pam")


def run_abdomen_case(
    ct_path: Path, seg_path: Path, anatomy: str, passive: str = PASSIVE_RECON
) -> dict[str, object]:
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
        passive_reconstruction=passive,
    )


def run_brain_case(passive: str = PASSIVE_RECON) -> dict[str, object]:
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
        passive_reconstruction=passive,
    )


# ── Hemispherical phased-array scaffold helper ─────────────────────────────────

def _draw_array_scaffold(
    ax: "plt.Axes",
    focus: np.ndarray,
    skin: np.ndarray,
    aperture_radius_m: float,
    n_lat: int = 6,
    n_lon: int = 16,
    n_arc: int = 80,
) -> None:
    """Render the mechanical scaffold that holds the skin-coupled phased-array
    elements as a sparse wireframe of latitude / longitude arcs.

    The modeled applicator is a **phased array of discrete piston transducers
    mounted on a hemispherical scaffold**, not a continuous radiating bowl
    surface. Drawing a solid translucent cap overstates the radiating area and
    visually conflates the scaffold geometry with a classical FUS bowl. The
    correct depiction is the scaffold support structure — a few latitude rings
    and a few longitude arcs — with the discrete element scatter (drawn
    elsewhere) showing where the actual piezo elements sit.

    Geometry: each scaffold point lies on the sphere of radius R centred at
    the focus F with axis d̂ = (F − S) / ‖F − S‖ pointing into the body:

        P(θ, φ) = F − R · [cos(θ) · d̂ + sin(θ) · (cos(φ) · ê₁ + sin(φ) · ê₂)]

    where θ ∈ [θ_cutout, θ_max] covers the active spherical cap (central
    cutout θ_cutout ≈ 10° for a coaxial imaging probe; rim θ_max ≈ 55°
    matching the Rust kwavers constant `BOWL_THETA_MAX_RAD`) and
    (ê₁, ê₂) is a Gram–Schmidt frame perpendicular to d̂.
    """
    # Match the Rust constants in `abdominal3d::bowl`.
    THETA_CUTOUT = 0.175  # ≈ 10°
    THETA_MAX = 0.960     # ≈ 55°

    focal_depth = float(np.linalg.norm(focus - skin))
    if focal_depth < 1e-6:
        return

    d_hat = (focus - skin) / focal_depth
    arb = np.array([0.0, 0.0, 1.0]) if abs(d_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d_hat, arb)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d_hat, e1)
    R = aperture_radius_m

    scaffold_color = "#90a4ae"  # cool grey, suggestive of metal scaffold
    scaffold_kwargs = dict(c=scaffold_color, lw=0.5, alpha=0.45, zorder=2)

    # Latitude rings: theta = const, phi sweeps 0..2π.
    phi_dense = np.linspace(0.0, 2.0 * np.pi, n_arc)
    for theta in np.linspace(THETA_CUTOUT, THETA_MAX, n_lat):
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        X = focus[0] - R * (cos_th * d_hat[0] + sin_th * (np.cos(phi_dense) * e1[0] + np.sin(phi_dense) * e2[0]))
        Y = focus[1] - R * (cos_th * d_hat[1] + sin_th * (np.cos(phi_dense) * e1[1] + np.sin(phi_dense) * e2[1]))
        Z = focus[2] - R * (cos_th * d_hat[2] + sin_th * (np.cos(phi_dense) * e1[2] + np.sin(phi_dense) * e2[2]))
        ax.plot(X, Y, Z, **scaffold_kwargs)

    # Longitude arcs: phi = const, theta sweeps θ_cutout..θ_max.
    theta_dense = np.linspace(THETA_CUTOUT, THETA_MAX, n_arc)
    for phi in np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False):
        cos_th = np.cos(theta_dense)
        sin_th = np.sin(theta_dense)
        X = focus[0] - R * (cos_th * d_hat[0] + sin_th * (np.cos(phi) * e1[0] + np.sin(phi) * e2[0]))
        Y = focus[1] - R * (cos_th * d_hat[1] + sin_th * (np.cos(phi) * e1[1] + np.sin(phi) * e2[1]))
        Z = focus[2] - R * (cos_th * d_hat[2] + sin_th * (np.cos(phi) * e1[2] + np.sin(phi) * e2[2]))
        ax.plot(X, Y, Z, **scaffold_kwargs)


# ── 3-D abdominal bowl rendering ────────────────────────────────────────────────

def render_abdominal_3d(
    geo: dict[str, object],
    anatomy_label: str,
    fig_path: Path,
) -> Path:
    """3-D scatter visualisation showing the skin-coupled hemispherical
    phased-array transducer on the skin surface.

    The body skin surface (with CT-bed voxels excluded by the largest-connected-
    component filter in Rust) and organ surface are shown as point clouds. The
    discrete array elements sit outside the body, visible on the skin. A
    sparse latitude/longitude wireframe depicts the mechanical scaffold the
    elements are mounted on. Beam lines connect elements to the organ
    centroid (focus). The skin contact point is marked with a circle.
    """
    body = np.asarray(geo["body_surface_points_m"], dtype=float)
    organ = np.asarray(geo["organ_surface_points_m"], dtype=float)
    elements = np.asarray(geo["therapy_elements_m"], dtype=float)
    starts = np.asarray(geo["beam_start_points_m"], dtype=float)
    ends = np.asarray(geo["beam_end_points_m"], dtype=float)
    focus = np.asarray(geo["focus_m"], dtype=float)
    skin = np.asarray(geo["skin_contact_m"], dtype=float)

    # Focal depth = skin-to-focus distance [mm]; scaffold radius = focal_depth/cos(θ_max).
    focal_depth_mm = 1e3 * float(np.linalg.norm(focus - skin))
    aperture_radius_mm = 1e3 * float(geo["transducer_radius_m"])

    fig = plt.figure(figsize=(14.0, 6.0), constrained_layout=True)
    fig.suptitle(
        f"Skin-coupled histotripsy focused bowl (HistoSonics-like) on {anatomy_label} CT — "
        f"{int(geo['element_count'])} discrete elements + central imaging window, "
        f"focal depth {focal_depth_mm:.0f} mm, radius of curvature {aperture_radius_mm:.0f} mm",
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
        # Mechanical scaffold — sparse lat/long wireframe showing the
        # hemispherical support structure that holds the discrete array
        # elements. Not a continuous radiating surface. Drawn first so it
        # appears behind the body and elements.
        _draw_array_scaffold(ax, focus, skin, float(geo["transducer_radius_m"]))

        # Patient skin surface — translucent but clearly legible so the viewer
        # can see the array elements sitting ON the skin (outside the body), not
        # inside the patient. The array elements are on the camera side (azimuth
        # is set to the bowl axis), so they render in front of the skin cloud and
        # remain visible despite the higher skin opacity. CT-bed voxels were
        # dropped by the largest-connected-component filter on the Rust side.
        if body.size:
            ax.scatter(
                body[:, 0], body[:, 1], body[:, 2],
                s=1.3, c="#8090a0", alpha=0.18, depthshade=False, rasterized=True,
                label="patient skin surface",
            )
        # Organ surface — amber.
        if organ.size:
            ax.scatter(
                organ[:, 0], organ[:, 1], organ[:, 2],
                s=3.0, c="#f5a623", alpha=0.55, depthshade=False,
                label=f"{anatomy_label} surface",
            )
        # Discrete array elements — red, outside the body (larger markers).
        if elements.size:
            ax.scatter(
                elements[:, 0], elements[:, 1], elements[:, 2],
                s=8.0, c="#d0021b", alpha=0.85, depthshade=False,
                label=f"array elements ({int(geo['element_count'])})",
                zorder=4,
            )
        # Scaffold legend entry — proxy Line2D (add_patch not supported on Axes3D).
        import matplotlib.lines as mlines  # noqa: PLC0415
        ax._scaffold_proxy = mlines.Line2D(  # stored so legend can reference it
            [], [], color="#90a4ae", lw=1.0, alpha=0.8,
            label="hemispherical scaffold (lat/long wireframe)",
        )

        # Skin contact point — lime circle (aperture vertex on skin).
        ax.scatter(
            [skin[0]], [skin[1]], [skin[2]],
            marker="o", s=120, c="lime", edgecolors="black", linewidths=0.8,
            zorder=6, label="skin contact (aperture vertex)",
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

    # Oblique view: camera on the array side, elevated for depth.
    ax_a.view_init(elev=elev_primary, azim=azim_primary)
    ax_a.set_title(
        "Oblique view — histotripsy focused bowl on skin surface\n"
        "(red discrete elements outside body, lime = skin contact, cyan = organ focus)",
        fontsize=9,
    )
    handles, labels = ax_a.get_legend_handles_labels()
    handles.insert(0, ax_a._scaffold_proxy)
    labels.insert(0, ax_a._scaffold_proxy.get_label())
    ax_a.legend(handles, labels, loc="lower left", fontsize=6, frameon=True)

    # Side view: 90° from aperture axis to show profile.
    ax_b.view_init(elev=15, azim=azim_side)
    ax_b.set_title(
        "Side view — array profile\n"
        "discrete elements on hemispherical scaffold sit outside skin; beams converge at organ centroid",
        fontsize=9,
    )

    _save_figure(fig, fig_path)
    plt.close(fig)
    return fig_path


# ── Brain focused-bowl rendering (calvarium-level) ──────────────────────────────

def render_brain_focused_bowl_3d(geo: dict[str, object], fig_path: Path) -> Path:
    """Render the 1024-element transcranial focused bowl covering the calvarium.

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
        f"Transcranial hemispherical helmet (InsightEC-like) over calvarium — "
        f"{int(geo['element_count'])} elements, "
        f"radius of curvature {1e3 * float(geo['bowl_radius_m']):.0f} mm, "
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
                s=1.0, c="#b8c0cc", alpha=0.18, depthshade=False, rasterized=True,
                label="scalp surface",
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
        (liver_result, "liver (skin-coupled bowl)", "#f5a623"),
        (kidney_result, "kidney (skin-coupled bowl)", "#7b68ee"),
        (brain_result, "brain (transcranial bowl)", "#4a9edd"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5), constrained_layout=True)
    fig.suptitle("Simulated pressure exposure — CT-derived heterogeneous media", fontsize=12)

    for ax, (result, label, _color) in zip(axes, cases):
        exposure = np.asarray(result["exposure"], dtype=float)
        lesion_target = np.asarray(result.get("lesion_target", np.zeros_like(exposure)), dtype=float)
        ct_raw = np.asarray(result["placement_ct_hu"], dtype=float)
        placement_body = np.asarray(
            result.get("placement_body_mask", np.ones_like(ct_raw, dtype=bool)),
            dtype=bool,
        )
        ct = (
            np.where(placement_body, ct_raw, -1000.0)
            if placement_body.shape == ct_raw.shape else ct_raw
        )
        solver_body = np.asarray(
            result.get("body_mask", np.ones_like(exposure, dtype=bool)),
            dtype=bool,
        )
        if solver_body.shape != exposure.shape:
            solver_body = np.ones_like(exposure, dtype=bool)
        target = np.asarray(result.get("target_mask", np.zeros_like(exposure)), dtype=bool)
        solver_body = _target_connected_body_mask(solver_body, target)
        therapy_body = _therapy_display_mask(solver_body, target)
        spacing = tuple(float(v) for v in result["placement_spacing_m"])
        extent = _image_extent_xy(ct, spacing)
        solver_extent = _solver_extent_in_placement_frame(result) or extent

        ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300, alpha=0.55)
        if str(result.get("anatomy", "")).lower() in {"liver", "kidney"}:
            source_pressure = float(result.get("source_pressure_pa", 0.0))
            exposure_abs = np.where(therapy_body, np.maximum(lesion_target, 0.0) * source_pressure, 0.0)
        else:
            exposure_abs = np.where(therapy_body, np.abs(exposure), 0.0)
        exposure_display = np.ma.masked_where(~therapy_body, exposure_abs)
        im = ax.imshow(
            exposure_display.T, cmap="inferno", origin="lower", extent=solver_extent,
            vmin=0.0, vmax=float(np.max(exposure_abs)) or 1.0, alpha=0.80,
        )
        # Target contour (solver-grid frame).
        if target.any():
            x = np.linspace(solver_extent[0], solver_extent[1], target.shape[0])
            y = np.linspace(solver_extent[2], solver_extent[3], target.shape[1])
            ax.contour(x, y, target.T.astype(float), levels=[0.5], colors="white", linewidths=0.9)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

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


# ── Image-then-treat sequence (per-anatomy) ────────────────────────────────────

HISTOTRIPSY_INTRINSIC_THRESHOLD_PA = 26.0e6
FD_STENCIL_BOUNDARY_HALO_CELLS = 2


def _masked_positive(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != image.shape:
        mask = np.ones_like(image, dtype=bool)
    return np.where(mask, np.maximum(np.asarray(image, dtype=float), 0.0), 0.0)


def _normalised_masked_display(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ma.MaskedArray, float]:
    if mask.shape != image.shape:
        mask = np.ones_like(image, dtype=bool)
    positive = _masked_positive(image, mask)
    peak = float(np.max(positive)) if positive.size else 0.0
    scale = peak if peak > 0.0 else 1.0
    display = np.ma.masked_where(~mask, positive / scale)
    return display, peak


def _equal_area_threshold(
    image: np.ndarray,
    target: np.ndarray,
    body: np.ndarray,
) -> float | None:
    values = np.asarray(image, dtype=float)
    target_count = int(np.count_nonzero(target & body))
    if target_count == 0:
        return None
    in_body = values[body]
    if in_body.size < target_count:
        return None
    return float(np.partition(in_body, in_body.size - target_count)[in_body.size - target_count])


def _pressure_label(pressure_pa: float) -> str:
    pressure_pa = float(pressure_pa)
    if abs(pressure_pa) >= 1.0e6:
        return f"{pressure_pa / 1.0e6:.2f} MPa"
    if abs(pressure_pa) >= 1.0e3:
        return f"{pressure_pa / 1.0e3:.1f} kPa"
    return f"{pressure_pa:.1f} Pa"


def _target_connected_body_mask(body: np.ndarray, target: np.ndarray) -> np.ndarray:
    body_mask = np.asarray(body, dtype=bool)
    target_mask = np.asarray(target, dtype=bool)
    if body_mask.shape != target_mask.shape or not np.any(body_mask) or not np.any(target_mask):
        return body_mask

    seeds = np.argwhere(body_mask & target_mask)
    if seeds.size == 0:
        target_centroid = np.mean(np.argwhere(target_mask), axis=0)
        body_points = np.argwhere(body_mask)
        seed = body_points[np.argmin(np.sum((body_points - target_centroid) ** 2, axis=1))]
    else:
        seed = seeds[0]

    connected = np.zeros_like(body_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque([(int(seed[0]), int(seed[1]))])
    nx, ny = body_mask.shape
    while q:
        ix, iy = q.popleft()
        if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
            continue
        if connected[ix, iy] or not body_mask[ix, iy]:
            continue
        connected[ix, iy] = True
        q.extend(((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)))
    return connected


def _erode_body_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    eroded = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        if not np.any(eroded):
            return eroded
        north = np.zeros_like(eroded)
        south = np.zeros_like(eroded)
        west = np.zeros_like(eroded)
        east = np.zeros_like(eroded)
        north[1:, :] = eroded[:-1, :]
        south[:-1, :] = eroded[1:, :]
        west[:, 1:] = eroded[:, :-1]
        east[:, :-1] = eroded[:, 1:]
        eroded = eroded & north & south & west & east
    return eroded


def _therapy_display_mask(body: np.ndarray, target: np.ndarray) -> np.ndarray:
    body_mask = _target_connected_body_mask(body, target)
    interior = _erode_body_mask(body_mask, FD_STENCIL_BOUNDARY_HALO_CELLS)
    target_mask = np.asarray(target, dtype=bool)
    if target_mask.shape == interior.shape:
        interior = interior | target_mask
    return interior if np.any(interior) else body_mask


def render_image_then_treat_sequence(
    result: dict[str, object],
    anatomy_label: str,
    fig_path: Path,
) -> Path:
    """Reconstructed-image-then-lesion-generation sequence for one anatomy.

    Four panels left → right:

    1. **CT slice** — the planning anatomy as acquired, with the target
       segmentation contoured in cyan.
    2. **Anatomy reconstruction (multi-angle imaging transmit)** —
       `anatomy_reconstruction`, the coherently-summed multi-angle
       finite-frequency reconstruction of the patient-specific
       sound-speed contrast. This is the imaging product of the
       transmit/receive imaging pulse sequence that immediately precedes
       therapy; it is the same-aperture analogue of plane-wave
       compounding (every imaging transmit is a broadband finite-frequency
       row of the same Born operator that the therapy elements would
       transmit through). White contour: planning target.
    3. **Fused lesion guidance** — `fused_reconstruction`, the normalized
       active/passive harmonic lesion-localization score that uses all
       same-aperture reconstruction channels. The red contour is the
       equal-area support threshold used by the Dice metric.
    4. **Focused therapy peak-pressure → predicted treatment support** —
       `exposure` (heterogeneous peak-pressure field after focused
       therapy transmit on the same applicator). Liver and kidney cases
       use the histotripsy intrinsic-threshold isoline (red, 26 MPa,
       Vlaisavljevich 2015) over the target-derived treatment support.
       The brain case uses the skull-corrected focus marker because this
       chapter's transcranial focused-bowl run uses a low-pressure
       focused-ultrasound exposure, not histotripsy.

    The figure makes the image-then-treat contract concrete: the same
    discrete-element aperture is used to acquire the imaging
    reconstruction (panel 2) and to deliver the lesion-forming exposure
    (panel 3); no separate diagnostic device is required.
    """
    ct_raw = np.asarray(result["placement_ct_hu"], dtype=float)
    anatomy_recon = np.asarray(result["anatomy_reconstruction"], dtype=float)
    fused_recon = np.asarray(result["fused_reconstruction"], dtype=float)
    exposure = np.asarray(result["exposure"], dtype=float)
    lesion_target = np.asarray(result.get("lesion_target", np.zeros_like(exposure)), dtype=float)
    target = np.asarray(result.get("target_mask", np.zeros_like(ct_raw)), dtype=bool)
    organ = np.asarray(result.get("organ_mask", np.zeros_like(target)), dtype=bool)
    spacing = tuple(float(v) for v in result["placement_spacing_m"])
    extent = _image_extent_xy(ct_raw, spacing)

    # Solver-grid arrays live in a body-bbox crop with its own origin.  Display
    # them at their TRUE physical bbox in placement coordinates instead of
    # stretching them onto the full-CT extent (the latter mislocates every solver
    # voxel by the crop offset, e.g. a recon pixel at solver (40,40) gets drawn
    # ~10 cm off-target).
    solver_extent = _solver_extent_in_placement_frame(result) or extent

    # Placement-grid body mask (full CT) — used to clip the displayed CT so the
    # bed/table strip outside the patient is hidden. Solver-grid body mask is
    # used to clip recon/exposure fields (same shape as `anatomy_recon`).
    placement_body = np.asarray(
        result.get("placement_body_mask", np.ones_like(ct_raw, dtype=bool)),
        dtype=bool,
    )
    if placement_body.shape == ct_raw.shape:
        ct = np.where(placement_body, ct_raw, -1000.0)
    else:
        ct = ct_raw

    solver_body = np.asarray(
        result.get("body_mask", np.ones_like(anatomy_recon, dtype=bool)),
        dtype=bool,
    )
    if solver_body.shape != anatomy_recon.shape:
        solver_body = np.ones_like(anatomy_recon, dtype=bool)
    solver_body = _target_connected_body_mask(solver_body, target)
    therapy_body = _therapy_display_mask(solver_body, target)

    def _contour_xy(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Solver-grid mask coordinates in placement frame.
        return (
            np.linspace(solver_extent[0], solver_extent[1], mask.shape[0]),
            np.linspace(solver_extent[2], solver_extent[3], mask.shape[1]),
        )

    def _overlay_anatomy_contours(ax: "plt.Axes", target_color: str) -> None:
        # Body outline (faint), organ outline (amber), target outline (anatomy-coloured).
        if solver_body.any():
            bx, by = _contour_xy(solver_body)
            ax.contour(bx, by, solver_body.T.astype(float),
                       levels=[0.5], colors="#8090a0", linewidths=0.5, alpha=0.55)
        if organ.any():
            ox, oy = _contour_xy(organ)
            ax.contour(ox, oy, organ.T.astype(float),
                       levels=[0.5], colors="#f5a623", linewidths=0.8, alpha=0.85)
        if target.any():
            tx, ty = _contour_xy(target)
            ax.contour(tx, ty, target.T.astype(float),
                       levels=[0.5], colors=target_color, linewidths=1.1)

    applicator = (
        "InsightEC-like hemispherical helmet"
        if anatomy_label == "brain"
        else "histotripsy focused bowl (HistoSonics-like)"
    )
    coupling = "transcranial" if anatomy_label == "brain" else "skin-coupled"
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 5.0), constrained_layout=True)
    fig.suptitle(
        f"{anatomy_label} — image-then-treat sequence on the same {coupling} "
        f"{applicator} (same-array transmit + passive-cavitation receive)",
        fontsize=11,
    )

    # ── Panel 1: planning CT slice ────────────────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(
        ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300,
    )
    _overlay_anatomy_contours(ax0, target_color="cyan")
    ax0.set_title("Planning CT slice + target / organ contour", fontsize=10)
    ax0.set_xlabel("x [m]", fontsize=8)
    ax0.set_ylabel("y [m]", fontsize=8)
    ax0.set_aspect("equal")

    # ── Panel 2: imaging reconstruction (image step) ──────────────────────────
    ax1 = axes[1]
    ax1.imshow(
        ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300, alpha=0.35,
    )
    # Clip recon to the body interior — off-body sidelobes are inversion
    # artefacts of the limited-aperture, monochromatic Born operator and add
    # no information about the patient anatomy.
    recon_display, recon_peak = _normalised_masked_display(np.abs(anatomy_recon), solver_body)
    im1 = ax1.imshow(
        recon_display.T, cmap="magma", origin="lower", extent=solver_extent,
        vmin=0.0, vmax=1.0,
    )
    # Force the recon panel axes to match the placement frame so spatial
    # comparison with panels 1 and 3 is direct.
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    _overlay_anatomy_contours(ax1, target_color="white")
    ax1.set_title(
        "Reconstructed image (linearised Born, single-pass H¹-Tikhonov)\n"
        f"in-body peak = {recon_peak:.3g} (a.u.), multi-frequency imaging transmit, same aperture",
        fontsize=9,
    )
    ax1.set_xlabel("x [m]", fontsize=8)
    ax1.set_aspect("equal")
    plt.colorbar(
        im1, ax=ax1, fraction=0.046, pad=0.02,
        label="|contrast| / in-body peak",
    )

    # ── Panel 3: fused lesion-localisation reconstruction ────────────────────
    ax2 = axes[2]
    ax2.imshow(
        ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300, alpha=0.35,
    )
    fused_display, fused_peak = _normalised_masked_display(fused_recon, solver_body)
    im2 = ax2.imshow(
        fused_display.T, cmap="viridis", origin="lower", extent=solver_extent,
        vmin=0.0, vmax=1.0,
    )
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    _overlay_anatomy_contours(ax2, target_color="white")
    threshold = _equal_area_threshold(_masked_positive(fused_recon, solver_body), target, solver_body)
    if threshold is not None and threshold > 0.0:
        fx, fy = _contour_xy(fused_recon)
        ax2.contour(
            fx, fy, _masked_positive(fused_recon, solver_body).T,
            levels=[threshold], colors="red", linewidths=1.2,
        )
    fusion_metrics = result.get("metrics", {}).get("fusion", {})
    ax2.set_title(
        "Fused lesion-localization image\n"
        f"peak = {fused_peak:.3g}, Dice = {float(fusion_metrics.get('dice_equal_area', 0.0)):.2f}, "
        f"CNR = {float(fusion_metrics.get('cnr', 0.0)):.1f}",
        fontsize=9,
    )
    ax2.set_xlabel("x [m]", fontsize=8)
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, label="fused score / peak")

    # ── Panel 4: focused therapy peak-pressure + predicted support ───────────
    ax3 = axes[3]
    ax3.imshow(
        ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300, alpha=0.50,
    )
    exposure_abs_full = np.abs(exposure)
    if anatomy_label.lower() in {"liver", "kidney"}:
        source_pressure = float(result.get("source_pressure_pa", 0.0))
        exposure_abs = np.where(therapy_body, np.maximum(lesion_target, 0.0) * source_pressure, 0.0)
    else:
        exposure_abs = np.where(therapy_body, exposure_abs_full, 0.0)
    exposure_peak = float(np.max(exposure_abs))
    exposure_peak_full = float(np.max(exposure_abs_full))
    if exposure_peak > 0.0:
        exposure_display = np.ma.masked_where(~therapy_body, exposure_abs)
        im3 = ax3.imshow(
            exposure_display.T, cmap="inferno", origin="lower", extent=solver_extent,
            vmin=0.0, vmax=exposure_peak, alpha=0.82,
        )
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02, label="peak pressure [Pa]")
        ex, ey = _contour_xy(exposure_abs)
        if anatomy_label.lower() in {"liver", "kidney"} and exposure_peak >= HISTOTRIPSY_INTRINSIC_THRESHOLD_PA:
            ex, ey = _contour_xy(exposure_abs)
            ax3.contour(
                ex, ey, exposure_abs.T,
                levels=[HISTOTRIPSY_INTRINSIC_THRESHOLD_PA],
                colors="red", linewidths=1.4,
            )
        elif anatomy_label.lower() == "brain":
            focus_xy = np.asarray(result.get("focus_m", result.get("placement_focus_m", ())), dtype=float)
            if focus_xy.size >= 2:
                ax3.scatter(
                    [float(focus_xy[0])],
                    [float(focus_xy[1])],
                    marker="+",
                    c="red",
                    s=60,
                    linewidths=1.5,
                    zorder=5,
                )
    _overlay_anatomy_contours(ax3, target_color="white")
    title_lines = [
        f"Focused-US treatment-support peak = {_pressure_label(exposure_peak)}",
    ]
    if anatomy_label.lower() in {"liver", "kidney"} and exposure_peak >= HISTOTRIPSY_INTRINSIC_THRESHOLD_PA:
        title_lines.append("red: intrinsic-threshold cavitation isoline (26 MPa)")
    elif anatomy_label.lower() == "brain":
        title_lines.append("red marker: skull-corrected focus target")
    else:
        title_lines.append(
            f"(< 26 MPa intrinsic threshold; raw-grid peak = {_pressure_label(exposure_peak_full)})"
        )
    ax3.set_title("\n".join(title_lines), fontsize=9)
    ax3.set_xlabel("x [m]", fontsize=8)
    ax3.set_aspect("equal")

    _save_figure(fig, fig_path)
    plt.close(fig)
    return fig_path


# ── Reconstruction metrics ─────────────────────────────────────────────────────

def render_reconstruction_metrics(
    liver_result: dict[str, object],
    kidney_result: dict[str, object],
    brain_result: dict[str, object],
    fig_path: Path,
    operator_results: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Bar chart of Dice (equal-area threshold) and CNR for each anatomy/channel.

    When ``operator_results`` (a mapping anatomy -> operator-mode result) is
    supplied, a third panel quantitatively compares the passive cavitation
    channels (subharmonic, ultraharmonic) reconstructed by the finite-frequency
    operator inverse versus genuine passive acoustic mapping (DMAS).
    """
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

    def _metric(result: dict[str, object], channel: str, key: str) -> float:
        return float(result.get("metrics", {}).get(channel, {}).get(key, 0.0))

    n_channels = len(channel_keys)
    x = np.arange(n_channels)
    width = 0.25
    colors = ["#f5a623", "#7b68ee", "#4a9edd"]

    have_comparison = operator_results is not None
    n_panels = 3 if have_comparison else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 5.0), constrained_layout=True)
    fig.suptitle(
        "Reconstruction fidelity — Dice (equal-area), CNR"
        + (", and passive operator-vs-PAM" if have_comparison else ""),
        fontsize=12,
    )

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

    # ── Panel 3: passive cavitation channels, operator vs PAM ────────────────
    if have_comparison:
        ax = axes[2]
        passive_bands = [("subharmonic", "f₀/2"), ("ultraharmonic", "3f₀/2")]
        labels: list[str] = []
        operator_dice: list[float] = []
        pam_dice: list[float] = []
        for result, anatomy_label in cases:
            op_result = operator_results.get(anatomy_label, {})
            for ch_key, band_label in passive_bands:
                labels.append(f"{anatomy_label}\n{band_label}")
                pam_dice.append(_metric(result, ch_key, "dice_equal_area"))
                operator_dice.append(_metric(op_result, ch_key, "dice_equal_area"))
        xx = np.arange(len(labels))
        bw = 0.4
        b_op = ax.bar(xx - bw / 2, operator_dice, bw, label="finite-freq operator", color="#9aa7b2")
        b_pam = ax.bar(xx + bw / 2, pam_dice, bw, label="genuine PAM (DMAS)", color="#d0021b", alpha=0.88)
        ax.bar_label(b_op, fmt="%.2f", fontsize=6, padding=1)
        ax.bar_label(b_pam, fmt="%.2f", fontsize=6, padding=1)
        ax.set_xticks(xx)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Dice (equal-area)", fontsize=9)
        ax.set_title(
            "Passive cavitation channels:\nfinite-frequency operator vs genuine PAM",
            fontsize=10,
        )
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7)

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
            "3-D clinical device geometry: skin-coupled abdominal focused bowls on liver/kidney, "
            "transcranial focused bowl at calvarium; fully simulated exposures and reconstructions "
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


def _solver_extent_in_placement_frame(result: dict[str, object]) -> list[float] | None:
    """Physical bbox of the solver-grid arrays (anatomy_reconstruction, exposure,
    body_mask) expressed in placement-frame coordinates (origin = full CT centre).

    The solver grid is a square crop of the source CT around the patient body,
    resampled from source-index range `[x0..=x1] × [y0..=y1]` (`crop_bounds_index`)
    of a `source_dimensions = (NX_src, NY_src)` CT at `source_spacing_m = (sx, sy)`
    onto a `grid_size × grid_size` solver grid. Returns `[x_min, x_max, y_min, y_max]`
    aligned to the placement extent so solver-grid arrays can be imshow'd at their
    true physical location instead of being stretched onto the placement extent.

    Returns None if the metadata is missing (back-compat with old result dicts).
    """
    bounds = result.get("crop_bounds_index")
    src_dims = result.get("source_dimensions")
    src_spacing = result.get("source_spacing_m")
    if bounds is None or src_dims is None or src_spacing is None:
        return None
    x0, x1, y0, y1 = (int(v) for v in bounds)
    nx_src, ny_src = (int(v) for v in src_dims)
    sx, sy = (float(v) for v in src_spacing)
    cx = 0.5 * (nx_src - 1)
    cy = 0.5 * (ny_src - 1)
    # Left edge of voxel x0 → right edge of voxel x1, in placement coordinates.
    return [
        (x0 - cx - 0.5) * sx,
        (x1 - cx + 0.5) * sx,
        (y0 - cy - 0.5) * sy,
        (y1 - cy + 0.5) * sy,
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

    # 2b. When showcasing genuine PAM, also run the finite-frequency operator
    # baseline so fig05 can quantitatively compare the passive channels.
    operator_results: dict[str, dict[str, object]] | None = None
    if PASSIVE_RECON == "pam":
        operator_results = {
            "liver": run_abdomen_case(LIVER_CT, LIVER_SEG, "liver", "operator"),
            "kidney": run_abdomen_case(KIDNEY_CT, KIDNEY_SEG, "kidney", "operator"),
            "brain": run_brain_case("operator"),
        }

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
            operator_results=operator_results,
        ),
        render_image_then_treat_sequence(
            liver_result, "liver",
            OUT_DIR / "fig06_liver_image_then_treat.png",
        ),
        render_image_then_treat_sequence(
            kidney_result, "kidney",
            OUT_DIR / "fig07_kidney_image_then_treat.png",
        ),
        render_image_then_treat_sequence(
            brain_result, "brain",
            OUT_DIR / "fig08_brain_image_then_treat.png",
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
