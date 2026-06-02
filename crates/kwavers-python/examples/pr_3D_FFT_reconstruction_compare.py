#!/usr/bin/env python3
"""
pr_3D_FFT_reconstruction_compare.py
======================================
Validates `time_reversal_reconstruction` in pykwavers by imaging a 3D
Gaussian initial-pressure sphere from a full-aperture planar sensor.

Physical setup
--------------
  Grid:   NX=32 × NY=32 × NZ=32, DX=1 mm (3-D)
  Medium: Homogeneous, lossless, c0=1500 m/s, ρ₀=1000 kg/m³
  Source: Gaussian IVP centred at (NX//2, NY//2, NZ//2), σ = 4 cells = 4 mm
  Sensor: Full planar array at i=0 (all NY×NZ = 1024 positions)
  PML:    20 cells outside (pml_inside=False)
  CFL:    0.20

Algorithm
---------
  time_reversal_reconstruction(sensor_data, sensor_positions, grid, c, fs)
  reconstructs the initial pressure field over the full (NX, NY, NZ) grid.

Parity criterion
----------------
  Pearson r between the reconstructed and true pressure fields on the
  center depth plane (x = NX//2) ≥ 0.85.
  Full-volume flat Pearson is reported for information.

Outputs
-------
  output/pr_3D_FFT_reconstruction_compare.png
  output/pr_3D_FFT_reconstruction_metrics.txt
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
NX, NY, NZ = 32, 32, 32
DX         = 1.0e-3      # [m] = 1 mm
C0         = 1500.0      # [m/s]
RHO0       = 1000.0      # [kg/m³]
SIGMA_IX   = 4           # Gaussian half-width [cells] = 4 mm
CFL        = 0.20
DT         = CFL * DX / C0   # ≈ 1.333e-7 s

X0_IX, Y0_IX, Z0_IX = NX // 2, NY // 2, NZ // 2

T_END = 2.2 * X0_IX * DX / C0  # 2.2× one-way travel time
NT    = int(np.ceil(T_END / DT)) + 10

TOL_PEARSON_PLANE = 0.85  # center depth plane
TOL_PEARSON_FLAT  = 0.70  # full volume (lower due to background artifacts)

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "pr_3D_FFT_reconstruction_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "pr_3D_FFT_reconstruction_metrics.txt"


def build_source() -> tuple[np.ndarray, np.ndarray]:
    sigma_m = SIGMA_IX * DX
    x_arr   = np.arange(NX) * DX
    y_arr   = np.arange(NY) * DX
    z_arr   = np.arange(NZ) * DX
    XX, YY, ZZ = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")
    x0_m = X0_IX * DX;  y0_m = Y0_IX * DX;  z0_m = Z0_IX * DX
    p0 = np.exp(
        -((XX - x0_m) ** 2 + (YY - y0_m) ** 2 + (ZZ - z0_m) ** 2) / sigma_m ** 2
    ).astype(np.float64)
    return p0, sigma_m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print(
        f"pr_3D_FFT_reconstruction:\n"
        f"  Grid {NX}×{NY}×{NZ}, dx={DX*1e3:.1f} mm, σ={SIGMA_IX*DX*1e3:.1f} mm\n"
        f"  Source ({X0_IX},{Y0_IX},{Z0_IX}), planar sensor at i=0\n"
        f"  CFL={CFL}, NT={NT}, DT={DT:.3e} s"
    )

    p0_3d, sigma_m = build_source()
    source = pkw.Source.from_initial_pressure(p0_3d)

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[0, :, :] = True
    n_sensor = int(sensor_mask.sum())
    sensor = pkw.Sensor.from_mask(sensor_mask)

    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    sim    = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(False)

    t0_sim  = time.perf_counter()
    result  = sim.run(time_steps=NT, dt=DT)
    t_sim   = time.perf_counter() - t0_sim
    print(f"  Simulation done in {t_sim:.1f} s")

    # sensor_data: (n_sensor, NT) — rows correspond to sensor positions in C order
    # Sensor is sensor_mask[0,:,:] → positions ordered j outer, k inner (C/row-major)
    p_data = np.asarray(result.sensor_data, dtype=np.float64)  # (NY*NZ, NT)
    p_ty   = p_data.T                                           # (NT, NY*NZ)

    # Build sensor position array matching C-order enumeration of (j, k) pairs
    jj, kk = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="ij")
    positions = np.zeros((n_sensor, 3))
    positions[:, 0] = 0.0
    positions[:, 1] = jj.ravel() * DX
    positions[:, 2] = kk.ravel() * DX

    fs = 1.0 / DT
    t0_recon = time.perf_counter()
    recon    = np.asarray(
        pkw.time_reversal_reconstruction(p_ty, positions, grid, C0, fs)
    )  # (NX, NY, NZ)
    t_recon = time.perf_counter() - t0_recon
    print(f"  Reconstruction done in {t_recon:.1f} s")

    # ---------------------------------------------------------------------------
    # Parity comparison
    # ---------------------------------------------------------------------------
    # Center depth plane: x = X0_IX (the source depth)
    plane_true  = p0_3d[X0_IX, :, :].ravel()
    plane_recon = recon[X0_IX, :, :].ravel()
    r_plane, _ = pearsonr(plane_true, plane_recon)

    r_flat, _  = pearsonr(p0_3d.ravel(), recon.ravel())

    passed = r_plane >= TOL_PEARSON_PLANE and r_flat >= TOL_PEARSON_FLAT
    status = "PASS" if passed else "FAIL"

    print(f"\n  3D TR reconstruction [{status}]:")
    print(f"    Center-plane Pearson r = {r_plane:.6f}  (≥ {TOL_PEARSON_PLANE})")
    print(f"    Full-volume  Pearson r = {r_flat:.6f}  (≥ {TOL_PEARSON_FLAT})")
    print(f"    Recon shape = {recon.shape}")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"3D Photoacoustic Reconstruction [{status}]\n"
        f"r_plane={r_plane:.4f} ≥ {TOL_PEARSON_PLANE}, "
        f"r_flat={r_flat:.4f} ≥ {TOL_PEARSON_FLAT}",
        fontsize=9,
    )

    for ax_row, field, label in [(axes[0], p0_3d, "True p₀"), (axes[1], recon, "Reconstructed")]:
        vmax = np.max(np.abs(field))
        for ax, (slc, title) in zip(ax_row, [
            (field[X0_IX, :, :], f"x-plane (i={X0_IX})"),
            (field[:, Y0_IX, :], f"y-plane (j={Y0_IX})"),
            (field[:, :, Z0_IX], f"z-plane (k={Z0_IX})"),
        ]):
            ax.imshow(slc.T, origin="lower", cmap="hot" if "True" in label else "RdBu_r",
                      vmin=0 if "True" in label else -vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{label}: {title}")

    fig.tight_layout()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURE_PATH}")
    print(f"parity_status: {status}")

    save_text_report(METRICS_PATH, "pr_3D_FFT_reconstruction_compare", [
        f"Status:               {status}",
        f"pearson_r_plane:      {r_plane:.6f}  (≥ {TOL_PEARSON_PLANE})",
        f"pearson_r_flat:       {r_flat:.6f}  (≥ {TOL_PEARSON_FLAT})",
        f"grid:                 {NX}x{NY}x{NZ}  dx={DX*1e3:.1f}mm",
        f"sigma_mm:             {sigma_m*1e3:.2f}",
        f"source_ix:            ({X0_IX},{Y0_IX},{Z0_IX})",
        f"n_sensor:             {n_sensor}",
        f"NT:                   {NT}",
        f"sim_s:                {t_sim:.2f}",
        f"recon_s:              {t_recon:.2f}",
    ])

    if not passed:
        if not args.allow_failure:
            sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
