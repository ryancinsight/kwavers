#!/usr/bin/env python3
"""
tvsp_slit_diffraction_compare.py
=================================
Validates the pykwavers PSTD solver against the Fraunhofer analytical
far-field diffraction pattern for a double-slit aperture.

Physical setup
--------------
  Grid:   NX=80 (propagation) × NY=100 (transverse) × NZ=1, dx=0.25 mm
  Medium: Homogeneous, lossless, c0=1500 m/s, ρ₀=1000 kg/m³
  Source: CW plane wave at f0=1 MHz (λ=1.5 mm) through two narrow slits
          Slit width: W = 3 cells = 0.75 mm = 0.5λ
          Slit separation (center-to-center): S = 10 cells = 2.5 mm = 1.67λ
          Slits centred at j=50 (y_c=12.5 mm), j_c1=45, j_c2=55
  Sensor: Line of sensors at x = 18.75 mm (i=75), recording all 100 y positions
  PML:    10 cells outside (pml_inside=False)

Analytical reference
--------------------
  Fraunhofer far-field diffraction pattern for coherent double slit:
    |P(θ)| = P₀ · |sinc(π·W·sin(θ)/λ) · cos(π·S·sin(θ)/λ)|
  where sinc is the normalised sinc, sin(θ) = Δy/√(L² + Δy²), Δy = y − y_c.

  Far-field criterion: L >> D²/λ  where D ≈ S + W = 3.25 mm
    D²/λ = 3.25²/1.5 ≈ 7 mm;  L = 18.75 mm ≈ 2.7×7 mm — valid.

Parity criterion
----------------
  Pearson r of normalised measured amplitude vs Fraunhofer prediction ≥ 0.95.

Outputs
-------
  output/tvsp_slit_diffraction_compare.png
  output/tvsp_slit_diffraction_metrics.txt
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
from scipy.stats import pearsonr

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
NX, NY, NZ = 80, 100, 1
DX = 0.25e-3       # [m] = 0.25 mm

C0   = 1500.0      # [m/s]
RHO0 = 1000.0      # [kg/m³]
F0   = 1.0e6       # [Hz]  1 MHz
LAMBDA = C0 / F0   # = 1.5 mm

# Slit geometry (in grid cells)
J_C    = NY // 2   # transverse centre = 50
SEP    = 10        # centre-to-centre separation [cells]
W      = 3         # slit width [cells]
# Slit 1: j ∈ [J_C - SEP//2 - W//2, J_C - SEP//2 + W//2]
# Slit 2: j ∈ [J_C + SEP//2 - W//2, J_C + SEP//2 + W//2]
J1_START = J_C - SEP // 2 - W // 2     # 44
J1_END   = J1_START + W                 # 47  (exclusive)
J2_START = J_C + SEP // 2 - W // 2     # 54
J2_END   = J2_START + W                 # 57  (exclusive)
J_C1 = (J1_START + J1_END - 1) / 2.0   # centre of slit 1
J_C2 = (J2_START + J2_END - 1) / 2.0   # centre of slit 2

SLIT_SEP_M = SEP * DX    # [m] = 2.5 mm
SLIT_W_M   = W   * DX    # [m] = 0.75 mm

# Source: CW at i=0 (left wall)
SOURCE_IX = 0

# Sensor: line at i=75 (5 cells before right wall → avoids PML edge)
SENSOR_IX = 75
L_M       = SENSOR_IX * DX   # propagation distance = 18.75 mm

# Time
CFL  = 0.25
DT   = CFL * DX / C0         # 4.17e-8 s
N_PERIODS    = 18             # total CW periods
N_LAST       = 3              # periods to average for steady-state amplitude
NT           = int(N_PERIODS / F0 / DT) + 1
N_LAST_STEPS = int(N_LAST / F0 / DT)

TOL_PEARSON = 0.95

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "tvsp_slit_diffraction_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_slit_diffraction_metrics.txt"


def fraunhofer_amp(y_arr: np.ndarray) -> np.ndarray:
    """
    Normalised Fraunhofer far-field amplitude for coherent double slit.

    |P(y)| / P_max = |sinc(π·W_m·sinθ/λ) · cos(π·S_m·sinθ/λ)|
    where sinc is the normalised sinc (sinc(x) = sin(πx)/(πx)).
    """
    y_c = J_C * DX        # physical centre [m]
    dy  = y_arr - y_c     # offset from centre [m]
    sin_theta = dy / np.sqrt(L_M ** 2 + dy ** 2)

    # Normalised sinc argument: x = W·sinθ/λ  (np.sinc(x) = sin(πx)/(πx))
    x_sinc = SLIT_W_M * sin_theta / LAMBDA

    # Interference phase argument: π·S·sinθ/λ
    x_cos = np.pi * SLIT_SEP_M * sin_theta / LAMBDA

    amp = np.abs(np.sinc(x_sinc) * np.cos(x_cos))
    return amp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print(
        f"tvsp_slit_diffraction:\n"
        f"  Grid {NX}×{NY}×{NZ}, dx={DX*1e3:.2f} mm, "
        f"f0={F0*1e-6:.0f} MHz, λ={LAMBDA*1e3:.1f} mm\n"
        f"  Slit W={SLIT_W_M*1e3:.2f} mm ({W} cells), "
        f"S={SLIT_SEP_M*1e3:.2f} mm ({SEP} cells), L={L_M*1e3:.1f} mm\n"
        f"  Slits: j∈[{J1_START},{J1_END}) and j∈[{J2_START},{J2_END})\n"
        f"  NT={NT}, dt={DT:.2e} s, averaging last {N_LAST} periods"
    )

    # -----------------------------------------------------------------------
    # Build source mask: CW at source column, slit positions only
    # -----------------------------------------------------------------------
    source_mask = np.zeros((NX, NY, NZ), dtype=np.float64)
    source_mask[SOURCE_IX, J1_START:J1_END, 0] = 1.0
    source_mask[SOURCE_IX, J2_START:J2_END, 0] = 1.0

    t_sig     = np.arange(NT) * DT
    cw_signal = np.sin(2.0 * np.pi * F0 * t_sig)

    source = pkw.Source.from_mask(source_mask, cw_signal, F0)

    # -----------------------------------------------------------------------
    # Sensor: transverse line at SENSOR_IX
    # -----------------------------------------------------------------------
    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[SENSOR_IX, :, 0] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    # -----------------------------------------------------------------------
    # Run PSTD
    # -----------------------------------------------------------------------
    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(False)

    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0
    print(f"  Simulation done in {elapsed:.1f} s")

    # -----------------------------------------------------------------------
    # Extract steady-state amplitude
    # -----------------------------------------------------------------------
    p_data = np.asarray(result.sensor_data, dtype=np.float64)  # (NY, NT)
    p_ss   = p_data[:, -N_LAST_STEPS:]                          # last N_LAST periods
    amp_sim = np.sqrt(2.0) * np.sqrt(np.mean(p_ss ** 2, axis=1))  # peak amplitude
    amp_sim_norm = amp_sim / (amp_sim.max() + 1e-30)

    # -----------------------------------------------------------------------
    # Analytical Fraunhofer reference
    # -----------------------------------------------------------------------
    y_arr = np.arange(NY) * DX
    amp_anal      = fraunhofer_amp(y_arr)
    amp_anal_norm = amp_anal / (amp_anal.max() + 1e-30)

    # -----------------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------------
    r, _ = pearsonr(amp_sim_norm, amp_anal_norm)
    passed = r >= TOL_PEARSON
    status = "PASS" if passed else "FAIL"

    print(f"\n  Fraunhofer comparison [{status}]:")
    print(f"  Status    : {status}")
    print(f"    Pearson r = {r:.6f}  (tolerance ≥ {TOL_PEARSON})")
    print(f"    Peak amplitude = {amp_sim.max():.4e} Pa")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Double-Slit Fraunhofer Diffraction [{status}]\n"
        f"f₀={F0*1e-6:.0f} MHz, W={SLIT_W_M*1e3:.2f} mm, "
        f"S={SLIT_SEP_M*1e3:.2f} mm, L={L_M*1e3:.1f} mm, Pearson={r:.4f}",
        fontsize=9,
    )

    y_mm = y_arr * 1e3
    ax = axes[0]
    ax.plot(y_mm, amp_sim_norm, "b-", lw=1.2, label="PSTD simulation")
    ax.plot(y_mm, amp_anal_norm, "r--", lw=1.2, label="Fraunhofer")
    ax.axvline(J_C * DX * 1e3, color="gray", lw=0.6, ls=":", label="grid center")
    ax.set_xlabel("Transverse position y [mm]")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("Amplitude at sensor line")
    ax.legend(fontsize=8)

    # Snapshot: pressure field at final step
    ax = axes[1]
    if hasattr(result, "pressure_field"):
        pf = np.asarray(result.pressure_field)[:, :, 0]
    else:
        pf = np.zeros((NX, NY))
    if pf.max() > 0:
        im = ax.imshow(pf.T, origin="lower", aspect="auto",
                       extent=[0, NX * DX * 1e3, 0, NY * DX * 1e3],
                       cmap="RdBu_r", vmin=-pf.max(), vmax=pf.max())
        ax.axvline(SENSOR_IX * DX * 1e3, color="g", lw=1.0, label="sensor")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title("Final pressure field")
        plt.colorbar(im, ax=ax, label="p [Pa]")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "(pressure field not recorded)", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURE_PATH}")

    save_text_report(METRICS_PATH, "tvsp_slit_diffraction_compare", [
        f"Status:             {status}",
        f"pearson_r:          {r:.6f}  (≥ {TOL_PEARSON})",
        f"slit_width_mm:      {SLIT_W_M*1e3:.3f}",
        f"slit_sep_mm:        {SLIT_SEP_M*1e3:.3f}",
        f"wavelength_mm:      {LAMBDA*1e3:.3f}",
        f"L_mm:               {L_M*1e3:.3f}",
        f"peak_amplitude_Pa:  {amp_sim.max():.4e}",
        f"NT:                 {NT}",
        f"runtime_s:          {elapsed:.2f}",
    ])

    if not passed:
        if not args.allow_failure:
            sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
