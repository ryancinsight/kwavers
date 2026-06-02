#!/usr/bin/env python3
"""
na_optimising_time_step_compare.py
===================================
Validates kwavers PSTD amplitude accuracy as a function of CFL number by
comparing the simulated peak pressure of a 1D Gaussian pulse against the
exact analytical solution.

Physical setup
--------------
  Grid:   NX=256 × NY=1 × NZ=1, DX=0.25 mm (PPW=12 at 500 kHz — grid error negligible)
  Medium: Homogeneous, lossless, c0=1500 m/s, ρ₀=1000 kg/m³
  Source: Gaussian IVP p0(x) = exp(−((x−x₀)/σ)²), x₀ = NX//2, σ = 6 mm
  Sensor: Single point at i = 200 (18 mm from centre)
  PML:    20 cells outside (pml_inside=False)

Analytical reference
--------------------
  1D free-space propagation splits the IVP into two half-amplitude copies:
    p_right(x,t) = 0.5 · p0(x − c0·t)
  Peak amplitude at sensor = 0.5 · A (A = 1 Pa normalised IVP).

CFL sweep
---------
  CFL ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
  DT = CFL · DX / c0;  NT = ceil(T_END / DT)

Parity criterion
----------------
  Amplitude error at CFL = 0.25 (kwavers default) ≤ 2 %.

Outputs
-------
  output/na_optimising_time_step_compare.png
  output/na_optimising_time_step_metrics.txt
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
NX, NY, NZ = 256, 1, 1
DX   = 0.25e-3     # [m] = 0.25 mm   → PPW = 12 at 500 kHz
C0   = 1500.0      # [m/s]
RHO0 = 1000.0      # [kg/m³]

SIGMA_M   = 6.0e-3  # [m] Gaussian half-width = 2 wavelengths @ 500 kHz
X0_IX     = NX // 2  # centre at i = 128
SENSOR_IX = 200      # 18 mm from centre

T_END     = 20.0e-6  # [s]  sufficient for peak + tail to pass sensor

# Reference amplitude: half the IVP peak (right-going wave)
AMP_ANAL = 0.5

CFL_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

# Parity gate: error at CFL = 0.25
CFL_REF   = 0.25
TOL_ERROR = 0.02   # 2 %

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "na_optimising_time_step_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_optimising_time_step_metrics.txt"


def run_cfl(cfl: float) -> float:
    """Run PSTD with given CFL; return peak sensor amplitude."""
    dt = cfl * DX / C0
    nt = int(np.ceil(T_END / dt))

    # Build IVP: Gaussian pressure, shape (NX, 1, 1)
    x_arr = np.arange(NX) * DX
    x0_m  = X0_IX * DX
    p0_1d = np.exp(-((x_arr - x0_m) / SIGMA_M) ** 2)
    p0_3d = p0_1d[:, np.newaxis, np.newaxis].astype(np.float64)

    source = pkw.Source.from_initial_pressure(p0_3d)

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[SENSOR_IX, 0, 0] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(False)

    result   = sim.run(time_steps=nt, dt=dt)
    p_data   = np.asarray(result.sensor_data, dtype=np.float64)
    return float(np.max(np.abs(p_data)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print(
        f"na_optimising_time_step:\n"
        f"  Grid {NX}×{NY}×{NZ}, dx={DX*1e3:.2f} mm, σ={SIGMA_M*1e3:.1f} mm\n"
        f"  Sensor i={SENSOR_IX} ({(SENSOR_IX-X0_IX)*DX*1e3:.1f} mm from centre)\n"
        f"  CFL sweep: {CFL_SWEEP}"
    )

    errors   = []
    peaks    = []
    runtimes = []

    for cfl in CFL_SWEEP:
        t0   = time.perf_counter()
        peak = run_cfl(cfl)
        elapsed = time.perf_counter() - t0
        err  = abs(peak - AMP_ANAL) / AMP_ANAL
        errors.append(err)
        peaks.append(peak)
        runtimes.append(elapsed)
        print(f"  CFL={cfl:.2f}: peak={peak:.5f} Pa, err={err*100:.3f}%, t={elapsed:.2f}s")

    # Parity check at CFL_REF
    idx_ref = CFL_SWEEP.index(CFL_REF)
    err_ref = errors[idx_ref]
    passed  = err_ref <= TOL_ERROR
    status  = "PASS" if passed else "FAIL"

    print(f"\n  [{status}] CFL={CFL_REF}: error={err_ref*100:.3f}%  (tolerance ≤ {TOL_ERROR*100:.0f}%)")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Time Step Optimisation — PSTD 1D Gaussian IVP [{status}]\n"
        f"Grid NX={NX}, dx={DX*1e3:.2f}mm, σ={SIGMA_M*1e3:.1f}mm, "
        f"L={( SENSOR_IX-X0_IX)*DX*1e3:.1f}mm",
        fontsize=9,
    )

    cfl_arr = np.array(CFL_SWEEP)
    err_arr = np.array(errors) * 100.0

    ax = axes[0]
    ax.semilogy(cfl_arr, err_arr, "b-o", lw=1.5, ms=5)
    ax.axvline(CFL_REF, color="r", lw=0.8, ls="--", label=f"CFL={CFL_REF} (default)")
    ax.axhline(TOL_ERROR * 100, color="g", lw=0.8, ls=":", label=f"{TOL_ERROR*100:.0f}% tolerance")
    ax.set_xlabel("CFL number")
    ax.set_ylabel("Amplitude error [%]")
    ax.set_title("Error vs CFL")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(cfl_arr, np.array(peaks), "b-o", lw=1.5, ms=5, label="Simulated peak")
    ax.axhline(AMP_ANAL, color="r", lw=1.0, ls="--", label=f"Analytical {AMP_ANAL:.2f} Pa")
    ax.set_xlabel("CFL number")
    ax.set_ylabel("Peak amplitude [Pa]")
    ax.set_title("Peak amplitude vs CFL")
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
        f"cfl_ref:              {CFL_REF}",
        f"error_at_cfl_ref_pct: {err_ref*100:.4f}  (≤ {TOL_ERROR*100:.0f}%)",
        f"analytical_peak_Pa:   {AMP_ANAL:.4f}",
    ]
    for i, cfl in enumerate(CFL_SWEEP):
        lines.append(
            f"cfl_{cfl:.2f}:  peak={peaks[i]:.5f} Pa, "
            f"err={errors[i]*100:.3f}%, t={runtimes[i]:.2f}s"
        )
    save_text_report(METRICS_PATH, "na_optimising_time_step_compare", lines)

    if not passed:
        if not args.allow_failure:
            sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
