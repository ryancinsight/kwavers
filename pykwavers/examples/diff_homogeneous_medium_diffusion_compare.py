#!/usr/bin/env python3
"""
diff_homogeneous_medium_diffusion_compare.py
============================================
Validates ``pykwavers.ThermalSimulation`` (Pennes bioheat transient) against
the exact analytical solution for a spatially uniform source in an
infinite/Neumann-BC domain.

Governing equation (no spatial gradients because Q and T are uniform):
  rho*cp * dT/dt = -wb*rho_b*cb*(T - Ta) + Q
Solution (with T(0) = Ta):
  T(t) = Ta + Q/(wb*rho_b*cb) * (1 - exp(-wb*rho_b*cb/(rho*cp) * t))

This is the exact Pennes ODE solution; the Laplacian term vanishes because
Q is spatially uniform so the temperature remains spatially uniform at all times.

Parity criteria (analytical reference):
  pearson_r >= 0.9999   (sub-0.01% deviation tolerated)
  rms_ratio in [0.999, 1.001]

Outputs
-------
  output/diff_homogeneous_medium_diffusion_compare.png
  output/diff_homogeneous_medium_diffusion_metrics.txt
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
# Physical parameters
# ---------------------------------------------------------------------------
NX, NY, NZ = 8, 8, 8          # small cube — spatial uniformity removes Laplacian
DX = DY = DZ = 0.5e-3         # [m]
RHO     = 1050.0               # tissue density [kg/m^3]
CP      = 3500.0               # specific heat [J/(kg*K)]
K_TH    = 0.52                 # thermal conductivity [W/(m*K)]
WB      = 0.10                 # blood perfusion [1/s]  (strong, tau ~= 35 s)
RHO_B   = 1050.0               # blood density [kg/m^3]
CP_B    = 3617.0               # blood specific heat [J/(kg*K)]
TA      = 37.0                 # arterial temperature [degC]
Q0      = 5.0e4                # uniform volumetric heat source [W/m^3]

DT      = 0.1                  # [s]
T_END   = 200.0                # [s]  (~ 5.7 time constants)
N_STEPS = int(T_END / DT)

PARITY_THRESHOLDS = {
    "pearson_r":     0.9999,
    "rms_ratio_min": 0.999,
    "rms_ratio_max": 1.001,
    "psnr_db":       60.0,
}

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "diff_homogeneous_medium_diffusion_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "diff_homogeneous_medium_diffusion_metrics.txt"

# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------
Omega_vol = WB * RHO_B * CP_B   # volumetric perfusion coefficient [W/(m^3*K)]
tau       = RHO * CP / Omega_vol  # perfusion time constant [s]
T_ss      = TA + Q0 / Omega_vol   # steady-state temperature [degC]

t_analytical = np.arange(N_STEPS) * DT
T_analytical = TA + (T_ss - TA) * (1.0 - np.exp(-t_analytical / tau))

print(f"Analytical: tau={tau:.2f} s, T_ss={T_ss:.4f} degC")

# ---------------------------------------------------------------------------
# pykwavers ThermalSimulation (uniform source, bioheat enabled)
# ---------------------------------------------------------------------------
Q_field = np.full((NX, NY, NZ), Q0, dtype=np.float64)
sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
sensor_mask[NX // 2, NY // 2, NZ // 2] = True  # center point

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
    sensor_mask=sensor_mask,
)

T_sim = np.asarray(result.temperature_at_sensors)[0, :]   # shape (N_STEPS,)

print(f"pykwavers:  T_final={T_sim[-1]:.4f} degC  (analytical={T_analytical[-1]:.4f})")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
metrics = compute_image_metrics(T_analytical, T_sim)

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
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Pennes Bioheat Transient -- Analytical vs pykwavers\n"
             f"Uniform Q={Q0:.0e} W/m^3, wb={WB} 1/s, tau={tau:.1f} s, T_ss={T_ss:.2f} degC",
             fontsize=10)

ax = axes[0]
ax.plot(t_analytical, T_analytical, "k--", lw=2, label="Analytical ODE")
ax.plot(t_analytical, T_sim, "C3-", lw=1.5, alpha=0.85, label="pykwavers ThermalSimulation")
ax.axhline(T_ss, color="gray", lw=0.8, ls=":", label=f"T_ss = {T_ss:.2f} degC")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [degC]")
ax.set_title("Temperature vs Time (center point)")
ax.legend()

ax = axes[1]
residual = T_sim - T_analytical
ax.plot(t_analytical, residual, "C0-", lw=1.0)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("T_sim - T_analytical [degC]")
ax.set_title(f"Residual  |  [{status}]  pearson={metrics['pearson_r']:.6f}  "
             f"rms_ratio={metrics['rms_ratio']:.6f}")

fig.tight_layout()
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_PATH}")

save_text_report(METRICS_PATH, "diff_homogeneous_medium_diffusion_compare", [
    f"Status:     {status}",
    f"pearson_r:  {metrics['pearson_r']:.8f}  (threshold >= {PARITY_THRESHOLDS['pearson_r']})",
    f"rms_ratio:  {metrics['rms_ratio']:.8f}  (threshold [{PARITY_THRESHOLDS['rms_ratio_min']}, {PARITY_THRESHOLDS['rms_ratio_max']}])",
    f"psnr_db:    {metrics['psnr_db']:.2f} dB  (threshold >= {PARITY_THRESHOLDS['psnr_db']} dB)",
    f"T_ss_analytical: {T_ss:.6f} degC",
    f"T_final_sim:     {T_sim[-1]:.6f} degC",
    f"tau_perfusion:   {tau:.4f} s",
    f"max_residual:    {np.max(np.abs(residual)):.6f} degC",
])

if not passed:
    sys.exit(1)
