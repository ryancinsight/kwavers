#!/usr/bin/env python3
"""
na_optimising_grid_parameters_compare.py
=========================================
Validates kwavers PSTD spectral accuracy as a function of points-per-wavelength
(PPW) by comparing the simulated peak pressure of a 1D Gaussian pulse against
the exact analytical solution.

Physical setup
--------------
  Physical domain: 80 mm (NX = round(0.08 / DX))
  Medium: Homogeneous, lossless, c0=1500 m/s, ρ₀=1000 kg/m³
  Reference frequency: f0 = 500 kHz  →  λ = 3 mm
  Grid spacing: DX = λ / PPW  for each PPW in sweep
  Source: Gaussian IVP p0(x) = exp(−((x−x₀)/σ)²),  σ = 2λ = 6 mm
  Sensor: 15 mm from Gaussian centre (right-going wave)
  PML:    20 cells outside (pml_inside=False)
  CFL:    fixed = 0.20

Analytical reference
--------------------
  1D free-space propagation:  peak sensor amplitude = 0.5 · A (A = 1 Pa IVP).

PPW sweep
---------
  PPW ∈ [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
  DX = λ / PPW;  NX = round(80 mm / DX);  CFL = 0.20 (fixed)

Parity criterion
----------------
  Amplitude error at PPW = 6 (minimum kwavers recommendation) ≤ 2 %.

Outputs
-------
  output/na_optimising_grid_parameters_compare.png
  output/na_optimising_grid_parameters_metrics.txt
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
# Fixed physical parameters
# ---------------------------------------------------------------------------
C0         = 1500.0    # [m/s]
RHO0       = 1000.0    # [kg/m³]
F0         = 500.0e3   # [Hz] reference frequency
LAMBDA     = C0 / F0   # [m] = 3 mm
SIGMA_M    = 2 * LAMBDA  # [m] Gaussian half-width = 6 mm
DOMAIN_M   = 80.0e-3  # [m] physical domain length
PROP_M     = 15.0e-3  # [m] propagation distance from Gaussian centre to sensor
T_END      = 20.0e-6  # [s] simulation end time (covers propagation + tail)
CFL_FIXED  = 0.20

AMP_ANAL   = 0.5  # right-going wave: half initial amplitude

PPW_SWEEP  = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
PPW_REF    = 6
TOL_ERROR  = 0.02  # 2 %

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "na_optimising_grid_parameters_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_optimising_grid_parameters_metrics.txt"


def run_ppw(ppw: int) -> tuple[float, int]:
    """Run PSTD at given PPW; return (peak amplitude, NX)."""
    dx   = LAMBDA / ppw
    nx   = int(round(DOMAIN_M / dx))
    dt   = CFL_FIXED * dx / C0
    nt   = int(np.ceil(T_END / dt))
    ny   = nz = 1

    x0_ix     = nx // 2
    sensor_ix = min(x0_ix + int(round(PROP_M / dx)), nx - 1)

    x_arr = np.arange(nx) * dx
    x0_m  = x0_ix * dx
    p0_1d = np.exp(-((x_arr - x0_m) / SIGMA_M) ** 2)
    p0_3d = p0_1d[:, np.newaxis, np.newaxis].astype(np.float64)

    source = pkw.Source.from_initial_pressure(p0_3d)

    sensor_mask = np.zeros((nx, ny, nz), dtype=bool)
    sensor_mask[sensor_ix, 0, 0] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    grid   = pkw.Grid(nx, ny, nz, dx, dx, dx)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(False)

    result = sim.run(time_steps=nt, dt=dt)
    p_data = np.asarray(result.sensor_data, dtype=np.float64)
    return float(np.max(np.abs(p_data))), nx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print(
        f"na_optimising_grid_parameters:\n"
        f"  c0={C0:.0f} m/s, f0={F0*1e-3:.0f} kHz, λ={LAMBDA*1e3:.1f} mm\n"
        f"  σ={SIGMA_M*1e3:.1f} mm, prop={PROP_M*1e3:.0f} mm, CFL={CFL_FIXED}\n"
        f"  PPW sweep: {PPW_SWEEP}"
    )

    errors   = []
    peaks    = []
    nxs      = []
    runtimes = []

    for ppw in PPW_SWEEP:
        t0 = time.perf_counter()
        peak, nx = run_ppw(ppw)
        elapsed  = time.perf_counter() - t0
        err  = abs(peak - AMP_ANAL) / AMP_ANAL
        errors.append(err)
        peaks.append(peak)
        nxs.append(nx)
        runtimes.append(elapsed)
        dx_mm = LAMBDA * 1e3 / ppw
        print(
            f"  PPW={ppw:2d}: dx={dx_mm:.3f} mm, NX={nx:4d}, "
            f"peak={peak:.5f} Pa, err={err*100:.3f}%, t={elapsed:.2f}s"
        )

    # Parity check at PPW_REF
    idx_ref = PPW_SWEEP.index(PPW_REF)
    err_ref = errors[idx_ref]
    passed  = err_ref <= TOL_ERROR
    status  = "PASS" if passed else "FAIL"

    print(f"\n  [{status}] PPW={PPW_REF}: error={err_ref*100:.3f}%  (tolerance ≤ {TOL_ERROR*100:.0f}%)")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Grid Parameter Optimisation — PSTD 1D Gaussian IVP [{status}]\n"
        f"c0={C0:.0f} m/s, f0={F0*1e-3:.0f} kHz, σ={SIGMA_M*1e3:.0f} mm, "
        f"CFL={CFL_FIXED}, PPW_ref={PPW_REF}, err={err_ref*100:.3f}%",
        fontsize=9,
    )

    ppw_arr = np.array(PPW_SWEEP)
    err_arr = np.array(errors) * 100.0

    ax = axes[0]
    ax.semilogy(ppw_arr, err_arr, "b-o", lw=1.5, ms=5)
    ax.axvline(PPW_REF, color="r", lw=0.8, ls="--", label=f"PPW={PPW_REF} (ref)")
    ax.axhline(TOL_ERROR * 100, color="g", lw=0.8, ls=":", label=f"{TOL_ERROR*100:.0f}% tolerance")
    ax.set_xlabel("Points per wavelength (PPW)")
    ax.set_ylabel("Amplitude error [%]")
    ax.set_title("Error vs PPW")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(ppw_arr, np.array(peaks), "b-o", lw=1.5, ms=5, label="Simulated peak")
    ax.axhline(AMP_ANAL, color="r", lw=1.0, ls="--", label=f"Analytical {AMP_ANAL:.2f} Pa")
    ax.set_xlabel("Points per wavelength (PPW)")
    ax.set_ylabel("Peak amplitude [Pa]")
    ax.set_title("Peak amplitude vs PPW")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURE_PATH}")
    print(f"parity_status: {status}")

    lines = [
        f"Status:               {status}",
        f"ppw_ref:              {PPW_REF}",
        f"error_at_ppw_ref_pct: {err_ref*100:.4f}  (≤ {TOL_ERROR*100:.0f}%)",
        f"analytical_peak_Pa:   {AMP_ANAL:.4f}",
        f"cfl_fixed:            {CFL_FIXED}",
        f"wavelength_mm:        {LAMBDA*1e3:.3f}",
        f"sigma_mm:             {SIGMA_M*1e3:.3f}",
    ]
    for i, ppw in enumerate(PPW_SWEEP):
        dx_mm = LAMBDA * 1e3 / ppw
        lines.append(
            f"ppw_{ppw:02d}: dx={dx_mm:.3f}mm NX={nxs[i]:4d} "
            f"peak={peaks[i]:.5f} Pa err={errors[i]*100:.3f}% t={runtimes[i]:.2f}s"
        )
    save_text_report(METRICS_PATH, "na_optimising_grid_parameters_compare", lines)

    if not passed:
        if not args.allow_failure:
            sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
