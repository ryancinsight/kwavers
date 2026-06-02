#!/usr/bin/env python3
"""
ewp_fluid_and_elastic_comparison_compare.py
=============================================
Reproduces the k-Wave MATLAB example_ewp_fluid_and_elastic_comparison.

Two acoustic PSTD simulations in a 2D (NZ=1) domain with a horizontal
fluid-solid interface:
  1. Fluid model  — homogeneous water everywhere (cs=0, cp=1500, rho=1000).
                    The "fluid approximation": no impedance mismatch at the
                    notional solid boundary.
  2. Elastic model (acoustic approximation) — heterogeneous acoustic medium:
                    water above the interface, solid acoustic properties
                    (cp=2000, rho=1200, cs=0) below.  Captures impedance-mismatch
                    reflection/transmission but omits shear-wave mode conversion.
                    Both models use the acoustic PSTD solver (SolverType.PSTD).

Note: The SolverType.Elastic PML instability (scalar isotropic damping that
explodes past NT≈200) was fixed in commit 5b2c5dd9. The fix replaces the scalar
PML with a separable per-axis exponential PML applied to both displacements and
velocities. A full elastic-with-shear comparison (cs>0 in solid, SolverType.Elastic)
is deferred until a k-Wave.jl reference for the elastic case is available.

A focused arc source (ring-IVP) in the fluid region concentrates energy on
the interface centre.  The fluid model shows a freely propagating focused
beam; the elastic model shows partial reflection at the interface and
transmission into the stiffer solid.

Physical setup
--------------
  Grid:       NX=128, NY=128, NZ=1, dx=dy=dz=0.5 mm
  PML:        10 cells, pml_inside=True
  Interface:  horizontal at j=INTERFACE_J=64
              Fluid: j < 64   cp=1500 m/s  rho=1000 kg/m³  (water)
              Solid: j >= 64  cp=2000 m/s  rho=1200 kg/m³  (bone-like, cs=0)
  Source:     Ring-IVP on arc of radius ARC_RADIUS=40 cells centred at the
              interface midpoint (64, 64).  The arc spans the upper semicircle
              (j < 64), producing a focused P-wave converging on the focal
              point at the fluid-solid boundary.
  Time:       NT=600, dt=5.0e-8 s → T_end = 30 µs
              P-wave convergence: 40×0.5mm/1500m/s ≈ 13.3 µs (step 267)
              Reflected/transmitted fronts clearly visible by step 500.

Validation criterion
--------------------
  Fluid-model focusing gain at focal point: p_max(focus) / p_mean(background)
  The focused ring-IVP must produce at least FOCUSING_GAIN_MIN × background.

Outputs
-------
  output/ewp_fluid_elastic_arc_mask.png       — binary arc source mask
  output/ewp_fluid_elastic_pmax_compare.png   — p_max dB fields side by side
  output/ewp_fluid_elastic_compare.txt        — numerical report

Usage::

    python examples/ewp_fluid_and_elastic_comparison_compare.py
    python examples/ewp_fluid_and_elastic_comparison_compare.py --allow-failure
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    save_text_report,
)

bootstrap_example_paths()
import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
NX, NY, NZ    = 128, 128, 1
DX            = 0.5e-3             # [m]

CP_FLUID      = 1500.0             # [m/s] P-wave speed, fluid (water)
CP_SOLID      = 2000.0             # [m/s] P-wave speed, solid (bone-like)
RHO_FLUID     = 1000.0             # [kg/m³] density, fluid
RHO_SOLID     = 1200.0             # [kg/m³] density, solid

PML           = 10
INTERFACE_J   = NY // 2            # j=64: fluid j<64, solid j≥64
FOCUS_I       = NX // 2            # 64
FOCUS_J       = INTERFACE_J        # 64

# Focused arc source: upper semicircle of radius ARC_RADIUS centred at focal point
ARC_RADIUS    = 40                 # cells = 20 mm
ARC_AMPLITUDE = 1.0                # normalised [Pa] — dB plots are relative to max

# Time-stepping: CFL = 0.3 based on max acoustic speed (solid)
# PSTD stability: CFL ≤ 0.3 per axis for pseudospectral 2D is conservative.
DT            = 0.3 * DX / CP_SOLID    # = 7.5e-8 s
NT            = 600                    # T_end = 45 µs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_arc_mask_2d() -> np.ndarray:
    """
    Binary mask (NX, NY) of the focused arc source.
    Upper semicircle of radius ARC_RADIUS centred at (FOCUS_I, FOCUS_J),
    restricted to the fluid region j < INTERFACE_J.
    Returns bool array shape (NX, NY).
    """
    ii = np.arange(NX)[:, np.newaxis]
    jj = np.arange(NY)[np.newaxis, :]
    dist = np.sqrt((ii - FOCUS_I) ** 2 + (jj - FOCUS_J) ** 2)
    on_arc   = np.abs(dist - ARC_RADIUS) <= 0.75
    in_fluid = jj < INTERFACE_J
    return (on_arc & in_fluid).astype(bool)


def run_pstd_model(
    arc_mask_2d: np.ndarray,
    cp_2d: np.ndarray,
    rho_2d: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    Acoustic PSTD simulation.

    Parameters
    ----------
    arc_mask_2d : (NX, NY) bool, active source cells.
    cp_2d       : (NX, NY) float, P-wave speed map.
    rho_2d      : (NX, NY) float, density map.
    label       : descriptive name for progress output.

    Returns
    -------
    p_max_field : (NX, NY, NZ) float, peak pressure over all time steps [Pa].
    """
    # Build medium — elastic_heterogeneous with cs=0 works for acoustic PSTD.
    cp_3d  = cp_2d[:, :, np.newaxis]  * np.ones((NX, NY, NZ))
    cs_3d  = np.zeros((NX, NY, NZ))
    rho_3d = rho_2d[:, :, np.newaxis] * np.ones((NX, NY, NZ))
    medium = pkw.Medium.elastic_heterogeneous(cp_3d, cs_3d, rho_3d)

    # Ring-IVP: initial pressure concentrated on the arc
    p0 = np.zeros((NX, NY, NZ))
    p0[:, :, 0] = arc_mask_2d.astype(np.float64) * ARC_AMPLITUDE
    source = pkw.Source.from_initial_pressure(p0)

    # Single-point sensor at focal point; p_max_field is tracked full-grid.
    sens = np.zeros((NX, NY, NZ), dtype=bool)
    sens[FOCUS_I, FOCUS_J, 0] = True
    sensor = pkw.Sensor.from_mask(sens)
    sensor.set_record(["p_max"])

    sim = pkw.Simulation(
        pkw.Grid(NX, NY, NZ, DX, DX, DX),
        medium, source, sensor,
        solver=pkw.SolverType.PSTD,
    )
    sim.set_pml_size(PML)
    sim.set_pml_inside(True)

    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0

    p_field = np.asarray(result.p_max_field, dtype=np.float64)
    print(f"  {label}: {elapsed:.1f} s  p_max peak = {np.abs(p_field).max():.4e} Pa")
    return p_field


def to_db(field_3d: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    """Convert absolute field (NX, NY, NZ) to dB normalised to its own peak."""
    f = np.abs(field_3d[:, :, 0])
    peak = f.max()
    if peak < 1e-30:
        return np.full_like(f, floor_db)
    return np.clip(20.0 * np.log10(f / peak + 1e-30), floor_db, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    arc_mask_2d = make_arc_mask_2d()
    n_arc = int(arc_mask_2d.sum())

    print(
        f"ewp_fluid_elastic_comparison:\n"
        f"  Grid {NX}×{NY}×{NZ}, dx={DX*1e3:.1f} mm, PML={PML}\n"
        f"  Interface j={INTERFACE_J}  focus=({FOCUS_I},{FOCUS_J})\n"
        f"  Fluid:  cp={CP_FLUID:.0f} m/s  rho={RHO_FLUID:.0f} kg/m³\n"
        f"  Solid:  cp={CP_SOLID:.0f} m/s  rho={RHO_SOLID:.0f} kg/m³  (cs=0, acoustic approx)\n"
        f"  Arc: radius={ARC_RADIUS} cells  n_active={n_arc}\n"
        f"  NT={NT}  dt={DT:.3e} s  T_end={NT*DT*1e6:.1f} µs"
    )

    # ------------------------------------------------------------------
    # Figure 1: Arc source mask
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.imshow(
        arc_mask_2d.T.astype(np.uint8),
        cmap="gray", origin="upper",
        extent=[0, NX * DX * 1e3, NY * DX * 1e3, 0],
        vmin=0, vmax=1,
    )
    ax1.axhline(INTERFACE_J * DX * 1e3, color="white", lw=1.0, ls="--", alpha=0.7)
    ax1.scatter(
        [FOCUS_I * DX * 1e3], [FOCUS_J * DX * 1e3],
        c="red", s=60, marker="+", zorder=5, label="focal point",
    )
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_title(f"Arc Source Mask (radius={ARC_RADIUS} cells, {n_arc} cells)")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    mask_path = out_dir / "ewp_fluid_elastic_arc_mask.png"
    plt.savefig(mask_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {mask_path}")

    # ------------------------------------------------------------------
    # Fluid model: homogeneous water everywhere (no impedance mismatch)
    # ------------------------------------------------------------------
    print("\nRunning fluid model (homogeneous water, acoustic PSTD) ...")
    cp_fluid_2d  = np.full((NX, NY), CP_FLUID)
    rho_fluid_2d = np.full((NX, NY), RHO_FLUID)
    p_max_fluid  = run_pstd_model(arc_mask_2d, cp_fluid_2d, rho_fluid_2d, "Fluid model")

    # ------------------------------------------------------------------
    # Elastic model (acoustic approximation): heterogeneous cp/rho, cs=0
    # ------------------------------------------------------------------
    print("\nRunning elastic model (acoustic approx: heterogeneous cp/rho, PSTD) ...")
    jj2 = np.arange(NY)[np.newaxis, :]   # (1, NY)
    cp_elast_2d  = np.where(jj2 < INTERFACE_J, CP_FLUID,  CP_SOLID)  * np.ones((NX, NY))
    rho_elast_2d = np.where(jj2 < INTERFACE_J, RHO_FLUID, RHO_SOLID) * np.ones((NX, NY))
    p_max_elast  = run_pstd_model(arc_mask_2d, cp_elast_2d, rho_elast_2d, "Elastic model")

    # ------------------------------------------------------------------
    # Figure 2: dB p_max comparison
    # ------------------------------------------------------------------
    FLOOR_DB   = -40.0
    db_fluid   = to_db(p_max_fluid,  FLOOR_DB)
    db_elast   = to_db(p_max_elast,  FLOOR_DB)
    extent_mm  = [0, NX * DX * 1e3, NY * DX * 1e3, 0]

    fig2, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, db_field, title in [
        (axes[0], db_fluid.T,  "Fluid Model"),
        (axes[1], db_elast.T,  "Elastic Model (acoustic approx)"),
    ]:
        im = ax.imshow(
            db_field,
            cmap="hot", origin="upper",
            extent=extent_mm,
            vmin=FLOOR_DB, vmax=0.0,
        )
        ax.axhline(INTERFACE_J * DX * 1e3, color="cyan", lw=1.0, ls="--", alpha=0.7)
        ax.scatter(
            [FOCUS_I * DX * 1e3], [FOCUS_J * DX * 1e3],
            c="cyan", s=40, marker="+", zorder=5,
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="dB re max", fraction=0.046, pad=0.04)

    fig2.suptitle(
        f"ewp_fluid_elastic_comparison: {NX}×{NY} grid, dx={DX*1e3:.1f}mm, "
        f"NT={NT}, dt={DT:.2e}s  (dashed cyan = interface)",
        fontsize=9,
    )
    fig2.tight_layout()
    compare_path = out_dir / "ewp_fluid_elastic_pmax_compare.png"
    plt.savefig(compare_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {compare_path}")

    # ------------------------------------------------------------------
    # Validation: focusing gain at focal point in fluid model
    # ------------------------------------------------------------------
    p_at_focus = float(np.abs(p_max_fluid[FOCUS_I, FOCUS_J, 0]))
    # Background: corner of valid domain, far from the beam
    p_bg = float(np.abs(p_max_fluid[PML:PML+5, PML:PML+5, 0]).mean() + 1e-30)
    focusing_gain = p_at_focus / p_bg

    FOCUSING_GAIN_MIN = 3.0
    passed = focusing_gain >= FOCUSING_GAIN_MIN
    status = "PASS" if passed else "FAIL"

    # Reflection coefficient at interface (normal incidence acoustic)
    Z_f  = RHO_FLUID * CP_FLUID
    Z_s  = RHO_SOLID * CP_SOLID
    R    = (Z_s - Z_f) / (Z_s + Z_f)
    T    = 2 * Z_s / (Z_s + Z_f)
    p_at_focus_elast = float(np.abs(p_max_elast[FOCUS_I, FOCUS_J, 0]))

    print(
        f"\nFocusing gain [{status}]: p_max(focus) = {p_at_focus:.4e} Pa  "
        f"p_mean(bg) = {p_bg:.4e} Pa  gain = {focusing_gain:.2f}×  "
        f"(threshold ≥ {FOCUSING_GAIN_MIN}×)\n"
        f"Interface: Z_fluid={Z_f:.3e}  Z_solid={Z_s:.3e}  "
        f"R={R:.3f}  T={T:.3f}\n"
        f"Elastic p_max(focus) = {p_at_focus_elast:.4e} Pa  "
        f"(expected ratio T≈{T:.2f})"
    )

    save_text_report(
        out_dir / "ewp_fluid_elastic_compare.txt",
        "ewp_fluid_elastic_comparison_compare",
        [
            f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, PML={PML}",
            f"Interface: j={INTERFACE_J} (fluid j<{INTERFACE_J}, solid j>={INTERFACE_J})",
            f"Fluid:  cp={CP_FLUID:.0f} m/s  rho={RHO_FLUID:.0f} kg/m3",
            f"Solid:  cp={CP_SOLID:.0f} m/s  rho={RHO_SOLID:.0f} kg/m3  (cs=0, acoustic approx)",
            f"Arc: radius={ARC_RADIUS} cells  n_active={n_arc}",
            f"NT={NT}  dt={DT:.4e} s  T_end={NT*DT*1e6:.2f} us",
            f"Z_fluid={Z_f:.4e}  Z_solid={Z_s:.4e}",
            f"Normal-incidence reflection coeff R={R:.4f}  transmission T={T:.4f}",
            f"Fluid p_max(focus) = {p_at_focus:.4e} Pa",
            f"Elastic p_max(focus) = {p_at_focus_elast:.4e} Pa",
            f"Focusing gain = {focusing_gain:.4f}x (threshold >= {FOCUSING_GAIN_MIN}x)",
            f"Status: {status}",
        ],
    )

    if not passed and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
