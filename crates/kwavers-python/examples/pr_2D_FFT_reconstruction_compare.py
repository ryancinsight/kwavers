#!/usr/bin/env python3
"""
pr_2D_FFT_reconstruction_compare.py
=====================================
Validates the pykwavers k-space line reconstruction (`kspace_line_recon`)
against a known 2D Gaussian initial pressure source.

Physical setup
--------------
  Grid:   NX=64 (depth) × NY=128 (aperture) × NZ=1, DX=0.5 mm
  Medium: Homogeneous, lossless, c0=1500 m/s, ρ₀=1000 kg/m³
  Source: Gaussian IVP centred at (NX//2, NY//2), σ = 5 cells = 2.5 mm
  Sensor: Full-aperture line at i=0 (all NY positions)
  PML:    20 cells outside (pml_inside=False)

Algorithm
---------
  kspace_line_recon(sensor_data.T, dy, dt, c, data_order='ty') maps the
  (NT, NY) sensor matrix back to an (NT, NY) image where depth index d
  corresponds to propagation distance x = d · DT · c0.

Analytical reference
--------------------
  True Gaussian y-profile at the source depth x0 = (NX//2) · DX.
  Expected depth row in reconstruction: row = round(x0 / c0 / DT).

Parity criterion
----------------
  Pearson r between reconstructed depth-slice profile and analytical
  Gaussian y-profile at source location ≥ 0.90.
  Max-over-depth Pearson ≥ 0.95.

Outputs
-------
  output/pr_2D_FFT_reconstruction_compare.png
  output/pr_2D_FFT_reconstruction_metrics.txt
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
NX, NY, NZ = 64, 128, 1
DX   = 0.5e-3      # [m] = 0.5 mm
C0   = 1500.0      # [m/s]
RHO0 = 1000.0      # [kg/m³]

X0_IX    = NX // 2          # source x-index = 32
Y0_IX    = NY // 2          # source y-index = 64
SIGMA_IX = 5                # Gaussian half-width [cells] = 2.5 mm

CFL  = 0.20
DT   = CFL * DX / C0        # ≈ 6.67e-8 s

# Simulate long enough to capture waves from source to sensor and partially reflect
T_END = 2.2 * X0_IX * DX / C0   # 2.2× one-way travel time
NT    = int(np.ceil(T_END / DT)) + 20

TOL_PEARSON_SLICE = 0.90   # at exact depth row
TOL_PEARSON_PROJ  = 0.95   # max-over-depth projection

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "pr_2D_FFT_reconstruction_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "pr_2D_FFT_reconstruction_metrics.txt"


def build_source() -> tuple[np.ndarray, np.ndarray]:
    """Return (p0_3d, p0_2d) — initial pressure IVP."""
    sigma_m = SIGMA_IX * DX
    x_arr   = np.arange(NX) * DX
    y_arr   = np.arange(NY) * DX
    XX, YY  = np.meshgrid(x_arr, y_arr, indexing="ij")
    x0_m    = X0_IX * DX
    y0_m    = Y0_IX * DX
    p0_2d   = np.exp(-((XX - x0_m) ** 2 + (YY - y0_m) ** 2) / sigma_m ** 2)
    p0_3d   = p0_2d[:, :, np.newaxis].astype(np.float64)
    return p0_3d, p0_2d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print(
        f"pr_2D_FFT_reconstruction:\n"
        f"  Grid {NX}×{NY}×{NZ}, dx={DX*1e3:.2f} mm, σ={SIGMA_IX*DX*1e3:.2f} mm\n"
        f"  Source at ({X0_IX},{Y0_IX}), sensor line at i=0\n"
        f"  CFL={CFL}, NT={NT}, DT={DT:.3e} s"
    )

    p0_3d, p0_2d = build_source()
    source = pkw.Source.from_initial_pressure(p0_3d)

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[0, :, 0] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    sim    = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(False)

    t0      = time.perf_counter()
    result  = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0
    print(f"  Simulation done in {elapsed:.1f} s")

    # sensor_data: (NY, NT) → transpose to (NT, NY) for kspace_line_recon
    p_data = np.asarray(result.sensor_data, dtype=np.float64)  # (NY, NT)
    p_ty   = p_data.T                                           # (NT, NY)

    p_recon = np.asarray(
        pkw.kspace_line_recon(p_ty, DX, DT, C0, data_order="ty")
    )  # (NT_recon, NY)

    # ---------------------------------------------------------------------------
    # Parity comparison: depth slice at expected source location
    # ---------------------------------------------------------------------------
    x0_m      = X0_IX * DX
    row_src   = int(round(x0_m / C0 / DT))
    row_src   = min(row_src, p_recon.shape[0] - 1)

    true_y    = p0_2d[X0_IX, :]          # analytical Gaussian y-profile
    recon_y   = p_recon[row_src, :]      # reconstructed depth-slice
    recon_abs = np.abs(p_recon)

    r_slice, _ = pearsonr(true_y, recon_y)
    r_proj, _  = pearsonr(true_y, np.max(recon_abs, axis=0))

    passed  = r_slice >= TOL_PEARSON_SLICE and r_proj >= TOL_PEARSON_PROJ
    status  = "PASS" if passed else "FAIL"

    print(f"\n  k-space line recon [{status}]:")
    print(f"    Depth-slice Pearson r = {r_slice:.6f}  (≥ {TOL_PEARSON_SLICE})")
    print(f"    Max-proj   Pearson r = {r_proj:.6f}  (≥ {TOL_PEARSON_PROJ})")
    print(f"    Row={row_src} (x={x0_m*1e3:.1f} mm), recon shape={p_recon.shape}")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"2D k-Space Line Reconstruction [{status}]\n"
        f"r_slice={r_slice:.4f} ≥ {TOL_PEARSON_SLICE}, "
        f"r_proj={r_proj:.4f} ≥ {TOL_PEARSON_PROJ}",
        fontsize=9,
    )

    y_mm = np.arange(NY) * DX * 1e3
    x_mm = np.arange(NX) * DX * 1e3

    ax = axes[0, 0]
    ax.imshow(
        p0_2d.T, origin="lower", aspect="auto",
        extent=[0, NX * DX * 1e3, 0, NY * DX * 1e3], cmap="hot"
    )
    ax.set_title("True initial pressure p₀")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

    ax = axes[0, 1]
    ext = [0, NT * DT * 1e6, 0, NY * DX * 1e3]
    vmax = np.max(np.abs(p_ty))
    ax.imshow(
        p_ty.T, origin="lower", aspect="auto", extent=ext,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax.set_title("Sensor data p(t, y)")
    ax.set_xlabel("t [µs]"); ax.set_ylabel("y [mm]")

    ax = axes[1, 0]
    vmax_r = np.max(np.abs(p_recon))
    ext_r = [0, p_recon.shape[0] * DT * C0 * 1e3, 0, NY * DX * 1e3]
    ax.imshow(
        p_recon.T, origin="lower", aspect="auto", extent=ext_r,
        cmap="RdBu_r", vmin=-vmax_r, vmax=vmax_r
    )
    ax.axvline(x0_m * 1e3, color="g", lw=0.8, label=f"source x={x0_m*1e3:.1f}mm")
    ax.set_title("Reconstructed image")
    ax.set_xlabel("depth [mm]"); ax.set_ylabel("y [mm]")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(y_mm, true_y / true_y.max(), "r--", lw=1.5, label="True p₀ y-profile")
    ax.plot(y_mm, recon_y / (np.abs(recon_y).max() + 1e-30), "b-", lw=1.2,
            label=f"Recon slice (row={row_src})")
    proj = np.max(recon_abs, axis=0)
    ax.plot(y_mm, proj / (proj.max() + 1e-30), "g:", lw=1.2,
            label=f"Recon max-proj (r={r_proj:.3f})")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title(f"Profile comparison (slice r={r_slice:.3f})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURE_PATH}")
    print(f"parity_status: {status}")

    save_text_report(METRICS_PATH, "pr_2D_FFT_reconstruction_compare", [
        f"Status:                {status}",
        f"pearson_r_slice:       {r_slice:.6f}  (≥ {TOL_PEARSON_SLICE})",
        f"pearson_r_proj:        {r_proj:.6f}  (≥ {TOL_PEARSON_PROJ})",
        f"source_x_mm:           {x0_m*1e3:.2f}",
        f"source_y_mm:           {Y0_IX*DX*1e3:.2f}",
        f"sigma_mm:              {SIGMA_IX*DX*1e3:.2f}",
        f"row_src:               {row_src}",
        f"recon_shape:           {list(p_recon.shape)}",
        f"NT:                    {NT}",
        f"runtime_s:             {elapsed:.2f}",
    ])

    if not passed:
        if not args.allow_failure:
            sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
