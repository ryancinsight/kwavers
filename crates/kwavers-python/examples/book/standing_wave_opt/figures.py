"""Figure rendering for Chapter 31 standing-wave suppression.

All inputs are plain Python dicts returned by `kw.run_standing_wave_suppression()`.
Numpy arrays in the dict are in C order with shape conventions matching the
Rust Array2 / Array3 layout: axis-0 = x (propagation), axis-1 = y (lateral).

Four figures are produced:

fig01_geometry.png       Sound-speed map with array, focus, reflective layer.
fig02_field_evolution.png  n_snapshots panels of |p|² in dB across iterations.
fig03_convergence.png    Three-panel time series: SWI, focal pressure, objective.
fig04_before_after.png   Initial vs final fields + focal-axis profiles + phases.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extent_mm(nx: int, ny: int, dx_m: float) -> list[float]:
    s = 1e3
    return [0.0, nx * dx_m * s, -0.5 * ny * dx_m * s, 0.5 * ny * dx_m * s]


def _db(re: np.ndarray, im: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    mag = np.sqrt(np.asarray(re, dtype=float) ** 2 + np.asarray(im, dtype=float) ** 2)
    peak = float(np.max(mag))
    if peak <= 0.0:
        return np.full(mag.shape, floor_db, dtype=float)
    ratio = np.maximum(mag / peak, 10.0 ** (floor_db / 20.0))
    return np.maximum(20.0 * np.log10(ratio), floor_db)


def _axial_profile(re: np.ndarray, im: np.ndarray, r: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_mm, I_norm) along focal axis between source and reflector."""
    x0 = r["source_x"]
    x1 = r["reflector_x_start"]
    hw = 2
    fy = r["focus_y"]
    ny = r["ny"]
    y0 = max(fy - hw, 0)
    y1 = min(fy + hw + 1, ny)
    intensity = np.mean(
        np.asarray(re[x0:x1, y0:y1], dtype=float) ** 2
        + np.asarray(im[x0:x1, y0:y1], dtype=float) ** 2,
        axis=1,
    )
    x_mm = np.arange(x0, x1) * r["dx_m"] * 1e3
    peak = float(np.max(intensity))
    return x_mm, intensity / (peak + 1e-30)


def _overlays(ax: plt.Axes, r: dict) -> None:
    dx_m = r["dx_m"]
    s = 1e3
    ax.axvline(r["reflector_x_start"] * dx_m * s, color="cyan", lw=0.9, ls="--", alpha=0.7)
    ax.axvline(r["reflector_x_end"] * dx_m * s, color="cyan", lw=0.9, ls="--", alpha=0.7)
    fx_mm = r["focus_x"] * dx_m * s
    fy_mm = (r["focus_y"] - 0.5 * r["ny"]) * dx_m * s
    ax.plot(fx_mm, fy_mm, "wx", ms=7, mew=1.5)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    except PermissionError:
        pass
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — geometry
# ---------------------------------------------------------------------------

def plot_geometry(r: dict, out_dir: Path) -> Path:
    dx_m = r["dx_m"]
    s = 1e3
    nx, ny = r["nx"], r["ny"]
    c_map = np.asarray(r["sound_speed_map"], dtype=float)
    ext = _extent_mm(nx, ny, dx_m)

    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    im = ax.imshow(c_map.T, cmap="bone", origin="lower", extent=ext,
                   vmin=1400.0, vmax=2200.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="sound speed [m/s]")

    lx0 = r["reflector_x_start"] * dx_m * s
    lx1 = r["reflector_x_end"] * dx_m * s
    ax.axvspan(lx0, lx1, color="cyan", alpha=0.18, label="reflective layer")

    eys = np.asarray(r["element_ys"], dtype=float)
    ex_mm = r["source_x"] * dx_m * s
    ey_mm = (eys - 0.5 * ny) * dx_m * s
    ax.scatter(np.full(len(ey_mm), ex_mm), ey_mm, marker="|", s=60,
               c="#e74c3c", zorder=5, label=f"{r['n_elements']} elements")

    ax.plot(r["focus_x"] * dx_m * s,
            (r["focus_y"] - 0.5 * ny) * dx_m * s,
            "wx", ms=10, mew=2.0, label="focal target", zorder=6)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"Chapter 31 — geometry: {r['n_elements']}-element array, "
        f"{r['frequency_hz']/1e3:.0f} kHz, dx={dx_m*1e3:.2f} mm\n"
        f"reflective layer R ≈ 0.32 (c_layer={r.get('c_layer_m_s', 2000):.0f} m/s)"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    path = out_dir / "fig01_geometry.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 2 — field snapshots across optimization iterations
# ---------------------------------------------------------------------------

def plot_field_evolution(r: dict, out_dir: Path) -> Path:
    snap_iters = np.asarray(r["snapshot_iterations"], dtype=int)
    snap_re = np.asarray(r["snapshot_fields_re"], dtype=float)
    snap_im = np.asarray(r["snapshot_fields_im"], dtype=float)
    swi_hist = np.asarray(r["swi_history"], dtype=float)
    n_snap = len(snap_iters)
    dx_m = r["dx_m"]
    ext = _extent_mm(r["nx"], r["ny"], dx_m)

    fig, axes = plt.subplots(1, n_snap, figsize=(3.2 * n_snap, 3.8), constrained_layout=True)
    if n_snap == 1:
        axes = [axes]
    fig.suptitle("Standing-wave suppression: field evolution across iterations", fontsize=10)

    for col, (it, gre, gim) in enumerate(zip(snap_iters, snap_re, snap_im)):
        ax = axes[col]
        db = _db(gre, gim)
        im = ax.imshow(db.T, cmap="inferno", origin="lower", extent=ext,
                       vmin=-40.0, vmax=0.0, aspect="auto")
        _overlays(ax, r)
        swi_it = float(swi_hist[it]) if it < len(swi_hist) else float(swi_hist[-1])
        ax.set_title(f"iter {it}\nSWI={swi_it:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_snap - 1:
            plt.colorbar(im, ax=ax, label="|p| [dB]", fraction=0.046, pad=0.02)

    path = out_dir / "fig02_field_evolution.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 3 — convergence time series
# ---------------------------------------------------------------------------

def plot_convergence(r: dict, out_dir: Path) -> Path:
    swi = np.asarray(r["swi_history"], dtype=float)
    pfocal = np.asarray(r["focal_pressure_history"], dtype=float)
    obj = np.asarray(r["objective_history"], dtype=float)
    iters = np.arange(len(swi))
    p_ref = float(r["focal_pressure_ref_pa"])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    fig.suptitle("Chapter 31 — standing-wave suppression convergence", fontsize=10)

    axes[0].plot(iters, swi, "o-", color="#e74c3c", ms=4, lw=1.5)
    axes[0].axhline(swi[0], ls="--", color="#e74c3c", alpha=0.4, lw=0.9,
                    label=f"initial {swi[0]:.3f}")
    axes[0].set(xlabel="iteration", ylabel="SWI", title="SWI reduction")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)

    pfocal_norm = pfocal / (p_ref + 1e-30)
    axes[1].plot(iters, pfocal_norm, "s-", color="#2e86de", ms=4, lw=1.5)
    axes[1].axhline(1.0, ls="--", color="#2e86de", alpha=0.4, lw=0.9, label="initial = 1.0")
    axes[1].set(xlabel="iteration", ylabel="normalised focal pressure",
                title="Focal pressure change")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)

    obj_norm = obj / (abs(obj[0]) + 1e-30)
    axes[2].plot(iters, obj_norm, "^-", color="#8e44ad", ms=4, lw=1.5)
    axes[2].axhline(0.0, ls="--", color="gray", alpha=0.4, lw=0.9)
    axes[2].set(xlabel="iteration", ylabel="normalised objective f(φ)",
                title="Objective convergence")
    axes[2].grid(True, alpha=0.25)

    path = out_dir / "fig03_convergence.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 4 — before / after comparison
# ---------------------------------------------------------------------------

def plot_before_after(r: dict, out_dir: Path) -> Path:
    dx_m = r["dx_m"]
    ext = _extent_mm(r["nx"], r["ny"], dx_m)
    swi = np.asarray(r["swi_history"], dtype=float)
    pfocal = np.asarray(r["focal_pressure_history"], dtype=float)

    pairs = (
        (r["initial_field_re"], r["initial_field_im"], "initial (DAS)", swi[0], pfocal[0]),
        (r["final_field_re"],   r["final_field_im"],   "optimized",     swi[-1], pfocal[-1]),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.0), constrained_layout=True)
    swi_red = 100.0 * (swi[0] - swi[-1]) / (swi[0] + 1e-12)
    pf_ratio = pfocal[-1] / (pfocal[0] + 1e-30)
    fig.suptitle(
        f"Chapter 31 — before / after:  SWI {swi[0]:.3f} → {swi[-1]:.3f} "
        f"(−{swi_red:.0f}%)   p_focal × {pf_ratio:.3f}",
        fontsize=10,
    )

    for col, (gre, gim, label, swi_v, pf_v) in enumerate(pairs):
        gre = np.asarray(gre, dtype=float)
        gim = np.asarray(gim, dtype=float)
        db = _db(gre, gim)

        ax = axes[0, col]
        im = ax.imshow(db.T, cmap="inferno", origin="lower", extent=ext,
                       vmin=-40.0, vmax=0.0, aspect="auto")
        _overlays(ax, r)
        ax.set_title(f"{label}\nSWI={swi_v:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="|p| [dB]", fraction=0.046, pad=0.02)

        ax = axes[1, col]
        x_mm, i_norm = _axial_profile(gre, gim, r)
        color = "#e74c3c" if col == 0 else "#2e86de"
        ax.plot(x_mm, i_norm, lw=1.5, color=color, label=label)
        ax.axvline(r["focus_x"] * dx_m * 1e3, ls="--", color="gold", lw=0.9, label="focus")
        ax.set(xlabel="x [mm]", ylabel="normalised intensity",
               title=f"Focal-axis profile — {label}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
        ax.set_ylim(bottom=0.0)

    # Phase comparison
    init_ph = np.degrees(np.asarray(r["initial_phases"], dtype=float))
    final_ph = np.degrees(np.asarray(r["final_phases"], dtype=float))
    elem_idx = np.arange(r["n_elements"])
    ax = axes[0, 2]
    ax.plot(elem_idx, init_ph, "o-", color="#e74c3c", ms=5, lw=1.5, label="initial DAS")
    ax.plot(elem_idx, final_ph, "s-", color="#2e86de", ms=5, lw=1.5, label="optimized")
    ax.set(xlabel="element index", ylabel="phase [°]",
           title="Element phases: DAS vs optimized")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    # Overlay comparison
    ax = axes[1, 2]
    for (gre, gim, label, *_), color in zip(pairs, ("#e74c3c", "#2e86de")):
        gre = np.asarray(gre, dtype=float)
        gim = np.asarray(gim, dtype=float)
        x_mm, i_norm = _axial_profile(gre, gim, r)
        swi_v = _[0]
        ax.plot(x_mm, i_norm, lw=1.5, color=color, label=f"{label} SWI={swi_v:.3f}")
    ax.axvline(r["focus_x"] * dx_m * 1e3, ls="--", color="gold", lw=0.9)
    ax.set(xlabel="x [mm]", ylabel="normalised intensity", title="Focal-axis profile overlay")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(bottom=0.0)

    path = out_dir / "fig04_before_after.png"
    _save(fig, path)
    return path
