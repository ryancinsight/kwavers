"""Figure and metrics writers for Chapter 32.

All renderers consume the result dicts returned by:
  - pykwavers.plan_abdominal_array_placement_from_ritk_ct (placement)
  - pykwavers.run_theranostic_inverse_from_ritk (result)

No physics is performed here.  This module is pure matplotlib rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers "3d" projection


# ── helpers ──────────────────────────────────────────────────────────────────

def _clamp_image(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=float), lo, hi)


def _extent_mm(array_2d: np.ndarray, spacing_m: float) -> list[float]:
    """Image extent in mm for matplotlib imshow."""
    nx, ny = array_2d.shape
    hx = 0.5 * nx * spacing_m * 1.0e3
    hy = 0.5 * ny * spacing_m * 1.0e3
    return [-hx, hx, -hy, hy]


def _contour_mask(ax: plt.Axes, mask: np.ndarray, extent: list[float], color: str, lw: float) -> None:
    if np.any(mask):
        ax.contour(
            np.asarray(mask, dtype=float).T,
            levels=[0.5],
            colors=[color],
            linewidths=lw,
            extent=extent,
            origin="lower",
        )


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# ── public renderers ─────────────────────────────────────────────────────────

def plot_3d_placement(placement: dict[str, object], path: Path) -> Path:
    """3-D scatter: HistoSonics bowl on skin surface targeting liver.

    Layout
    ------
    - Grey surface scatter: anterior body (skin) surface sampled from CT HU.
    - Red surface scatter: liver organ surface.
    - Blue scatter + wires: 256-element focused bowl element positions.
    - Gold dashed rays: per-element beam paths to the focus.
    - Gold star: focus (liver tumor centroid).
    - Cyan diamond: nearest skin contact point (transducer coupling face).

    Equal-aspect 3-D bounding cube is enforced to prevent cropping.
    """
    body_pts = np.asarray(placement["body_surface_points_m"]) * 1.0e3   # (N, 3) mm
    organ_pts = np.asarray(placement["organ_surface_points_m"]) * 1.0e3 # (M, 3) mm
    elem_pts = np.asarray(placement["therapy_elements_m"]) * 1.0e3      # (K, 3) mm
    beam_starts = np.asarray(placement["beam_start_points_m"]) * 1.0e3  # (B, 3) mm
    beam_ends = np.asarray(placement["beam_end_points_m"]) * 1.0e3      # (B, 3) mm
    focus = np.array(placement["focus_m"]) * 1.0e3                      # (3,) mm
    contact = np.array(placement["skin_contact_m"]) * 1.0e3             # (3,) mm

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample body surface to keep the scatter manageable
    step = max(1, len(body_pts) // 4000)
    ax.scatter(
        body_pts[::step, 0], body_pts[::step, 1], body_pts[::step, 2],
        s=2, c="#bdbdbd", alpha=0.25, label="body surface",
    )

    step_organ = max(1, len(organ_pts) // 1500)
    ax.scatter(
        organ_pts[::step_organ, 0], organ_pts[::step_organ, 1], organ_pts[::step_organ, 2],
        s=4, c="#e85d26", alpha=0.55, label="liver surface",
    )

    ax.scatter(
        elem_pts[:, 0], elem_pts[:, 1], elem_pts[:, 2],
        s=12, c="#1565c0", alpha=0.9, label=f"bowl elements ({len(elem_pts)})",
    )

    # Beam rays — subsample to avoid clutter
    n_rays = min(64, len(beam_starts))
    step_ray = max(1, len(beam_starts) // n_rays)
    for s, e in zip(beam_starts[::step_ray], beam_ends[::step_ray]):
        ax.plot(
            [s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
            color="#f9a825", alpha=0.22, linewidth=0.6,
        )

    ax.scatter(*focus, s=140, c="#f9a825", marker="*", zorder=10, label="focus (tumor centroid)")
    ax.scatter(*contact, s=90, c="#00bcd4", marker="D", zorder=10, label="skin contact")

    # Equal-aspect bounding cube
    all_pts = np.vstack([body_pts[::step], elem_pts, focus[None], contact[None]])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = max(float((hi - lo).max()) * 0.55, 1.0)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    synthetic_tag = " (synthetic phantom)" if placement.get("synthetic_phantom") else ""
    ax.set_title(
        f"HistoSonics 256-element bowl — liver placement{synthetic_tag}\n"
        f"R_f = {float(placement['transducer_radius_m']) * 1e3:.0f} mm · "
        f"{int(placement['element_count'])} elements on skin"
    )
    ax.legend(loc="upper left", fontsize=7, markerscale=1.4)
    ax.view_init(elev=28, azim=-55)

    return _save(fig, path)


def plot_exposure_slice(result: dict[str, object], path: Path) -> Path:
    """2-D cross-section: CT HU background + Westervelt peak-pressure exposure.

    Left panel  — CT HU (body windowed) with organ / target / body contours.
    Right panel — Westervelt peak-pressure exposure map (dB re max) with
                  organ and target contours overlaid.
    """
    spacing_m = float(result["spacing_m"])
    ct_hu = np.asarray(result["ct_hu"], dtype=float)
    body_mask = np.asarray(result["body_mask"], dtype=bool)
    organ_mask = np.asarray(result["organ_mask"], dtype=bool)
    target_mask = np.asarray(result["target_mask"], dtype=bool)
    exposure = np.asarray(result["exposure"], dtype=float)
    peak_pa = np.asarray(result["exposure_raw_peak_pressure"], dtype=float)

    extent = _extent_mm(ct_hu, spacing_m)
    exposure_db = 20.0 * np.log10(
        np.clip(np.abs(exposure), 1e-12, None) / max(float(np.max(np.abs(exposure))), 1e-12)
    )
    peak_db = 20.0 * np.log10(
        np.clip(peak_pa, 1e-12, None) / max(float(np.max(peak_pa)), 1e-12)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    # Left: CT anatomy
    win_ct = _clamp_image(ct_hu, -200.0, 400.0)
    axes[0].imshow(win_ct.T, origin="lower", extent=extent, cmap="gray", aspect="equal")
    _contour_mask(axes[0], body_mask, extent, "#ffffff", 0.7)
    _contour_mask(axes[0], organ_mask, extent, "#e85d26", 1.2)
    _contour_mask(axes[0], target_mask, extent, "#ffef5e", 1.5)
    axes[0].set_title("CT anatomy (HU −200→+400)")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    # Right: peak-pressure exposure
    im = axes[1].imshow(
        np.ma.masked_where(~body_mask, peak_db).T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=-30.0,
        vmax=0.0,
        aspect="equal",
    )
    _contour_mask(axes[1], organ_mask, extent, "#e85d26", 1.2)
    _contour_mask(axes[1], target_mask, extent, "#ffef5e", 1.5)
    axes[1].set_title("Westervelt peak pressure [dB re max]")
    axes[1].set_xlabel("x [mm]")
    fig.colorbar(im, ax=axes[1], shrink=0.82, label="dB")

    return _save(fig, path)


def plot_reconstructions(result: dict[str, object], path: Path) -> Path:
    """4-panel multi-modal reconstruction.

    Panels (left-to-right, top-to-bottom):
      1. Anatomy reconstruction  — acoustic-speed FWI image.
      2. Waveform RTM            — waveform-misfit reverse-time migration.
      3. Elastic-shear RTM       — ElasticPSTD shear-wave FWI channel.
      4. Fused reconstruction    — weighted multi-modal fusion.

    Each panel overlays organ (orange) and target (yellow) contours.
    """
    spacing_m = float(result["spacing_m"])
    organ_mask = np.asarray(result["organ_mask"], dtype=bool)
    target_mask = np.asarray(result["target_mask"], dtype=bool)

    panels = [
        ("anatomy_reconstruction", "Anatomy (acoustic-speed FWI)", "RdBu_r"),
        ("waveform_rtm_reconstruction", "Waveform RTM", "seismic"),
        ("elastic_shear_reconstruction", "Elastic-shear RTM", "PiYG"),
        ("fused_reconstruction", "Fused multi-modal", "magma"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (key, title, cmap) in zip(axes_flat, panels):
        arr = np.asarray(result[key], dtype=float)
        extent = _extent_mm(arr, spacing_m)
        peak = max(float(np.max(np.abs(arr))), 1e-12)
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=-peak,
            vmax=peak,
            aspect="equal",
        )
        _contour_mask(ax, organ_mask, extent, "#e85d26", 1.1)
        _contour_mask(ax, target_mask, extent, "#ffef5e", 1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.80)

    return _save(fig, path)


def plot_fwi_convergence(result: dict[str, object], path: Path) -> Path:
    """FWI objective-function convergence history.

    Plots the acoustic-speed / nonlinearity FWI objective history (outer
    iterations).  Adds a secondary inset for the elastic-shear objective
    history when available.
    """
    obj_hist = np.asarray(result["objective_history"], dtype=float)
    elastic_hist = result.get("elastic_shear_objective_history", [])
    elastic_hist = np.asarray(elastic_hist, dtype=float) if len(elastic_hist) > 0 else None

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)

    iters = np.arange(1, len(obj_hist) + 1)
    ax.semilogy(iters, obj_hist, marker="o", color="#1565c0", linewidth=1.8, label="FWI objective")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Objective (Charbonnier)")
    ax.set_title(
        f"FWI convergence — anatomy={result['anatomy']}  ·  "
        f"misfit={result['waveform_misfit']}  ·  "
        f"{len(obj_hist)} iterations"
    )
    ax.grid(True, which="both", alpha=0.3)

    if elastic_hist is not None and len(elastic_hist) > 1:
        ax2 = ax.twinx()
        ax2.semilogy(
            np.arange(1, len(elastic_hist) + 1),
            elastic_hist,
            marker="s",
            color="#c62828",
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
            label="elastic-shear FWI",
        )
        ax2.set_ylabel("Elastic-shear objective", color="#c62828")
        ax2.tick_params(axis="y", labelcolor="#c62828")
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    return _save(fig, path)


# ── metrics ───────────────────────────────────────────────────────────────────

def write_metrics(
    placement: dict[str, object],
    result: dict[str, object],
    figures: list[Path],
    path: Path,
) -> Path:
    """Write chapter metrics JSON.

    Parameters
    ----------
    placement:
        Dict from ``plan_abdominal_array_placement_from_ritk_ct``.
    result:
        Dict from ``run_theranostic_inverse_from_ritk``.
    figures:
        List of saved figure paths.
    path:
        Output JSON file path.
    """
    obj_hist = list(map(float, result["objective_history"]))
    elastic_hist = list(map(float, result.get("elastic_shear_objective_history", [])))

    payload: dict[str, object] = {
        "chapter": 32,
        "analysis": (
            "segmentation-driven HistoSonics liver histotripsy: "
            "3-D bowl placement + nonlinear FWI theranostic reconstruction"
        ),
        "placement": {
            "anatomy_label": placement.get("anatomy_label"),
            "element_count": placement.get("element_count"),
            "transducer_radius_m": float(placement.get("transducer_radius_m", 0.0)),
            "geometry_model": placement.get("geometry_model"),
            "synthetic_phantom": bool(placement.get("synthetic_phantom", True)),
            "focus_m": list(map(float, placement["focus_m"])),
            "skin_contact_m": list(map(float, placement["skin_contact_m"])),
        },
        "simulation": {
            "anatomy": result.get("anatomy"),
            "device_model": result.get("device_model"),
            "geometry_model": result.get("geometry_model"),
            "exposure_model": result.get("exposure_model"),
            "operator_model": result.get("operator_model"),
            "waveform_model": result.get("waveform_model"),
            "waveform_misfit": result.get("waveform_misfit"),
            "is_full_wave_inversion": result.get("is_full_wave_inversion"),
            "uses_nonlinear_wave_propagation": result.get("uses_nonlinear_wave_propagation"),
            "iterative_elastic_fwi": result.get("iterative_elastic_fwi"),
            "element_count": result.get("element_count"),
            "frequencies_hz": result.get("frequencies_hz"),
            "source_pressure_pa": result.get("source_pressure_pa"),
            "spacing_m": result.get("spacing_m"),
        },
        "convergence": {
            "fwi_iterations": len(obj_hist),
            "fwi_initial_objective": obj_hist[0] if obj_hist else None,
            "fwi_final_objective": obj_hist[-1] if obj_hist else None,
            "fwi_objective_history": obj_hist,
            "elastic_fwi_iterations": len(elastic_hist),
            "elastic_fwi_objective_history": elastic_hist,
        },
        "metrics": result.get("metrics", {}),
        "figures": [str(f) for f in figures],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
