#!/usr/bin/env python3
"""
diff_homogeneous_medium_source_compare.py
==========================================
Validates the spatial Laplacian and bioheat solver in ``ThermalSimulation``
against a pure Python forward-Euler Pennes reference on a 3D grid.

Both implementations share identical parameters, Q field, time step, and
initial conditions. Agreement must be within floating-point arithmetic
tolerances of the same algorithm executed in Python vs Rust.

Physical setup
--------------
  Grid:    16x16x32, dx=dy=dz=0.5 mm
  Medium:  homogeneous, soft tissue defaults
  Source:  Q(i,j,k) = Q0 * exp(-((i-cx)^2+(j-cy)^2+(k-cz)^2)/sigma^2)
           Gaussian focal heat source at grid centre
  Bioheat: wb=0.009 1/s (brain perfusion), Ta=37 degC
  Time:    DT=0.10 s, T_end=30 s (N_steps=300)

Comparison
----------
  Reference: forward-Euler in NumPy (identical to the Rust implementation)
  pykwavers: Rust ThermalDiffusionSolver via ThermalSimulation

Parity criteria (same-algorithm comparison):
  pearson_r  >= 0.9999
  rms_ratio  in [0.999, 1.001]
  psnr_db    >= 60 dB   (effectively identical fields)

Outputs
-------
  output/diff_homogeneous_medium_source_compare.png
  output/diff_homogeneous_medium_source_metrics.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()
import pykwavers as pkw

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NY, NZ = 16, 16, 32
DX = DY = DZ = 0.5e-3

RHO, CP, K_TH = 1040.0, 3650.0, 0.52
WB, RHO_B, CP_B, TA = 0.009, 1050.0, 3617.0, 37.0
Q0 = 3.0e5          # focal heat source amplitude [W/m^3]
SIGMA = 2.0         # Gaussian width [voxels]

DT     = 0.10
T_END  = 30.0
N_STEPS = int(T_END / DT)

PARITY_THRESHOLDS = {
    "pearson_r":     0.9999,
    "rms_ratio_min": 0.999,
    "rms_ratio_max": 1.001,
    "psnr_db":       60.0,
}

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "diff_homogeneous_medium_source_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "diff_homogeneous_medium_source_metrics.txt"

# ---------------------------------------------------------------------------
# Gaussian heat source field
# ---------------------------------------------------------------------------
cx, cy, cz = NX // 2, NY // 2, NZ // 2
I_idx, J_idx, K_idx = np.meshgrid(
    np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij"
)
r2 = (I_idx - cx)**2 + (J_idx - cy)**2 + (K_idx - cz)**2
Q_field = Q0 * np.exp(-r2 / SIGMA**2)

# ---------------------------------------------------------------------------
# Python forward-Euler Pennes reference
# ---------------------------------------------------------------------------
alpha_th = K_TH / (RHO * CP)                    # m^2/s
omega    = WB * RHO_B * CP_B / (RHO * CP)       # 1/s
Q_KS     = Q_field / (RHO * CP)                 # K/s

T_ref = np.full((NX, NY, NZ), TA, dtype=np.float64)

def laplacian_3d(T_arr, dr):
    """3D central-difference Laplacian matching Rust ThermalDiffusionSolver exactly.

    Rust iterates i in 1..nx-1, j in 1..ny-1, k in 1..nz-1 — boundary voxels
    remain at laplacian_workspace=0. Interior stencils still read boundary T values.
    """
    L = np.zeros_like(T_arr)
    dr2 = dr * dr
    ix = np.s_[1:-1, 1:-1, 1:-1]
    L[ix] = (
        (T_arr[2:, 1:-1, 1:-1] - 2.0 * T_arr[1:-1, 1:-1, 1:-1] + T_arr[:-2, 1:-1, 1:-1]) / dr2
        + (T_arr[1:-1, 2:, 1:-1] - 2.0 * T_arr[1:-1, 1:-1, 1:-1] + T_arr[1:-1, :-2, 1:-1]) / dr2
        + (T_arr[1:-1, 1:-1, 2:] - 2.0 * T_arr[1:-1, 1:-1, 1:-1] + T_arr[1:-1, 1:-1, :-2]) / dr2
    )
    return L

print(f"Running Python FD reference ({N_STEPS} steps)...")
for _ in range(N_STEPS):
    L = laplacian_3d(T_ref, DX)
    T_ref += DT * (alpha_th * L - omega * (T_ref - TA) + Q_KS)

print(f"Python FD done.  T_focus={T_ref[cx, cy, cz]:.4f} degC")

# ---------------------------------------------------------------------------
# pykwavers ThermalSimulation
# ---------------------------------------------------------------------------
sim = pkw.ThermalSimulation(
    nx=NX, ny=NY, nz=NZ,
    dx=DX, dy=DY, dz=DZ,
    thermal_conductivity=K_TH,
    density=RHO,
    specific_heat=CP,
    enable_bioheat=True,
    perfusion_rate=WB,
    blood_density=RHO_B,
    blood_specific_heat=CP_B,
    arterial_temperature=TA,
    metabolic_heat=0.0,
    initial_temperature=TA,
    track_thermal_dose=False,
)
result = sim.run(
    time_steps=N_STEPS,
    dt=DT,
    heat_source=Q_field,
)
T_pkw = np.asarray(result.temperature)    # (NX, NY, NZ)
print(f"pykwavers done. T_focus={T_pkw[cx, cy, cz]:.4f} degC")

# ---------------------------------------------------------------------------
# Metrics (full 3D field — Python reference matches Rust BC exactly)
# ---------------------------------------------------------------------------
metrics = compute_image_metrics(T_ref.ravel(), T_pkw.ravel())
passed = (
    metrics["pearson_r"]     >= PARITY_THRESHOLDS["pearson_r"]
    and PARITY_THRESHOLDS["rms_ratio_min"] <= metrics["rms_ratio"] <= PARITY_THRESHOLDS["rms_ratio_max"]
    and metrics["psnr_db"]   >= PARITY_THRESHOLDS["psnr_db"]
)
status = "PASS" if passed else "FAIL"
print(f"\nParity [{status}]  pearson={metrics['pearson_r']:.6f}  "
      f"rms_ratio={metrics['rms_ratio']:.6f}  psnr={metrics['psnr_db']:.1f} dB")
print(f"  Status    : {status}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    f"Spatial Laplacian Parity: Python FD vs pykwavers  [{status}]\n"
    f"NX={NX}, NY={NY}, NZ={NZ}, DX={DX*1e3:.1f} mm, DT={DT} s, T_end={T_END:.0f} s",
    fontsize=9,
)
iz_focus = cz
vmin = T_ref[:, :, iz_focus].min()
vmax = T_ref[:, :, iz_focus].max()

im = axes[0].imshow(T_ref[:, :, iz_focus].T, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
axes[0].set_title("Python FD reference  (z=centre)")
axes[0].set_xlabel("x [voxels]"); axes[0].set_ylabel("y [voxels]")
plt.colorbar(im, ax=axes[0], label="T [degC]")

im = axes[1].imshow(T_pkw[:, :, iz_focus].T, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
axes[1].set_title("pykwavers ThermalSimulation")
axes[1].set_xlabel("x [voxels]")
plt.colorbar(im, ax=axes[1], label="T [degC]")

diff2d = (T_pkw - T_ref)[:, :, iz_focus]
max_abs = float(np.max(np.abs(diff2d)))
im = axes[2].imshow(diff2d.T, origin="lower", cmap="RdBu_r", aspect="auto",
                    vmin=-max_abs, vmax=max_abs)
axes[2].set_title(f"Difference (pkw - ref)  max|diff|={max_abs:.2e} degC")
axes[2].set_xlabel("x [voxels]")
plt.colorbar(im, ax=axes[2], label="dT [degC]")

fig.tight_layout()
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_PATH}")

save_text_report(METRICS_PATH, "diff_homogeneous_medium_source_compare", [
    f"Status:      {status}",
    f"pearson_r:   {metrics['pearson_r']:.8f}  (threshold >= {PARITY_THRESHOLDS['pearson_r']})",
    f"rms_ratio:   {metrics['rms_ratio']:.8f}  (threshold [{PARITY_THRESHOLDS['rms_ratio_min']}, {PARITY_THRESHOLDS['rms_ratio_max']}])",
    f"psnr_db:     {metrics['psnr_db']:.2f} dB  (threshold >= {PARITY_THRESHOLDS['psnr_db']} dB)",
    f"max_abs_diff:{metrics['max_abs_diff']:.2e} degC",
    f"T_focus_ref: {T_ref[cx,cy,cz]:.6f} degC",
    f"T_focus_pkw: {T_pkw[cx,cy,cz]:.6f} degC",
])

if not passed:
    sys.exit(1)
