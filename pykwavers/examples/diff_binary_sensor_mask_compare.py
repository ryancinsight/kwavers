#!/usr/bin/env python3
"""
diff_binary_sensor_mask_compare.py
====================================
Validates that ``ThermalSimulation`` sensor recording correctly extracts
temperature time series at specified grid positions.

Strategy
--------
1. Run a thermal simulation with a spatially non-uniform source
   Q(i,j,k) = Q0 * (i+1)/(NX) * (j+1)/(NY) * (k+1)/(NZ)  (ramp field)
   and bioheat enabled (so each voxel reaches a distinct steady state).
2. Record sensor_data at 8 corner-adjacent interior points.
3. After the run, verify: sensor_data[s, -1] == temperature_field[si, sj, sk]
   (sensor final value matches field at same location).
4. Verify the sensor time series matches the analytical Pennes ODE for each
   voxel's local Q value:
     T_s(t) = Ta + Q(s)/(Omega_vol) * (1 - exp(-t/tau))

Parity criteria (sensor vs field):
  max absolute difference at final step: < 1e-6 degC  (floating-point identity)
  pearson_r vs analytical time series:   >= 0.9999
  rms_ratio vs analytical:               in [0.999, 1.001]

Outputs
-------
  output/diff_binary_sensor_mask_compare.png
  output/diff_binary_sensor_mask_metrics.txt
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
NX, NY, NZ = 12, 12, 12
DX = DY = DZ = 0.5e-3
RHO, CP, K_TH = 1050.0, 3500.0, 0.52
WB, RHO_B, CP_B, TA = 0.10, 1050.0, 3617.0, 37.0
Q0 = 6.0e4         # max heat source [W/m^3]

DT     = 0.10
T_END  = 150.0
N_STEPS = int(T_END / DT)

FIELD_TOL = 1e-4   # degC  -- sensor final value vs field final value

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "diff_binary_sensor_mask_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "diff_binary_sensor_mask_metrics.txt"

# ---------------------------------------------------------------------------
# Spatially varying Q field (ramp in i, j, k)
# ---------------------------------------------------------------------------
I_idx, J_idx, K_idx = np.meshgrid(
    np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij"
)
Q_field = Q0 * (I_idx + 1) / NX * (J_idx + 1) / NY * (K_idx + 1) / NZ

# ---------------------------------------------------------------------------
# Sensor positions: 8 interior points at 1/4 and 3/4 of domain in each axis
# ---------------------------------------------------------------------------
q1, q3 = NX // 4, 3 * NX // 4
sensor_positions = [
    (q1, q1, q1), (q1, q1, q3), (q1, q3, q1), (q1, q3, q3),
    (q3, q1, q1), (q3, q1, q3), (q3, q3, q1), (q3, q3, q3),
]
sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
for (si, sj, sk) in sensor_positions:
    sensor_mask[si, sj, sk] = True

# ---------------------------------------------------------------------------
# Run simulation
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
    sensor_mask=sensor_mask,
)

sensor_data = np.asarray(result.temperature_at_sensors)  # (n_sensors, N_STEPS)
T_field     = np.asarray(result.temperature)             # (NX, NY, NZ)

print(f"sensor_data shape: {sensor_data.shape}")
print(f"T_field shape:     {T_field.shape}")

# ---------------------------------------------------------------------------
# Test 1: sensor final value must match field at sensor position
# ---------------------------------------------------------------------------
Omega_vol = WB * RHO_B * CP_B
tau       = RHO * CP / Omega_vol

# Sensor order from thermal_bindings.rs (Fortran order: k outermost, i innermost):
sensor_positions_sorted = sorted(sensor_positions, key=lambda p: (p[2], p[1], p[0]))

field_at_sensors_final = np.array([
    T_field[si, sj, sk] for (si, sj, sk) in sensor_positions_sorted
])
sensor_final = sensor_data[:, -1]

max_field_diff = float(np.max(np.abs(sensor_final - field_at_sensors_final)))
print(f"\nTest 1 -- sensor vs field final value:")
print(f"  max |sensor - field|: {max_field_diff:.2e} degC  (tol={FIELD_TOL})")
test1_pass = max_field_diff < FIELD_TOL

# ---------------------------------------------------------------------------
# Test 2: sensor time series vs analytical Pennes ODE for each sensor Q
# ---------------------------------------------------------------------------
t_vec = np.arange(N_STEPS) * DT
pearson_vals = []
rms_ratio_vals = []

for s_idx, (si, sj, sk) in enumerate(sensor_positions_sorted):
    Q_local = float(Q_field[si, sj, sk])
    T_ss    = TA + Q_local / Omega_vol
    T_anal  = TA + (T_ss - TA) * (1.0 - np.exp(-t_vec / tau))
    m = compute_image_metrics(T_anal, sensor_data[s_idx, :])
    pearson_vals.append(m["pearson_r"])
    rms_ratio_vals.append(m["rms_ratio"])

pearson_min  = float(min(pearson_vals))
rms_rat_min  = float(min(rms_ratio_vals))
rms_rat_max  = float(max(rms_ratio_vals))
test2_pass   = pearson_min >= 0.9999 and 0.999 <= rms_rat_min and rms_rat_max <= 1.001
print(f"\nTest 2 -- sensor time series vs analytical ODE:")
print(f"  min pearson_r:  {pearson_min:.6f}  (threshold >= 0.9999)")
print(f"  rms_ratio range: [{rms_rat_min:.6f}, {rms_rat_max:.6f}]  (threshold [0.999, 1.001])")

overall_pass = test1_pass and test2_pass
status = "PASS" if overall_pass else "FAIL"
print(f"\nOverall [{status}]")
print(f"  Status    : {status}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(f"Sensor Mask Validation -- pykwavers ThermalSimulation  [{status}]", fontsize=10)

ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, 8))
for s_idx, (si, sj, sk) in enumerate(sensor_positions_sorted):
    Q_local = float(Q_field[si, sj, sk])
    T_ss_loc = TA + Q_local / Omega_vol
    T_anal = TA + (T_ss_loc - TA) * (1.0 - np.exp(-t_vec / tau))
    ax.plot(t_vec, T_anal, "--", color=colors[s_idx], lw=1.4, alpha=0.7)
    ax.plot(t_vec, sensor_data[s_idx, :], "-", color=colors[s_idx], lw=1.0,
            label=f"({si},{sj},{sk})")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [degC]")
ax.set_title("Sensor time series (solid=sim, dashed=analytical)")
ax.legend(fontsize=7, ncol=2)

ax = axes[1]
ax.bar(range(8), pearson_vals, color=colors)
ax.axhline(0.9999, color="k", lw=0.8, ls="--", label="threshold 0.9999")
ax.set_xticks(range(8))
ax.set_xticklabels([f"s{i}" for i in range(8)], rotation=45)
ax.set_ylabel("Pearson r vs analytical ODE")
ax.set_title(f"Per-sensor Pearson  |  min={pearson_min:.6f}")
ax.legend()
ax.set_ylim(0.998, 1.001)

fig.tight_layout()
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURE_PATH}")

save_text_report(METRICS_PATH, "diff_binary_sensor_mask_compare", [
    f"Status:           {status}",
    f"Test1 sensor==field: {'PASS' if test1_pass else 'FAIL'}  max_diff={max_field_diff:.2e} degC",
    f"Test2 pearson_min:   {pearson_min:.8f}  (threshold >= 0.9999)",
    f"Test2 rms_ratio:     [{rms_rat_min:.6f}, {rms_rat_max:.6f}]  (threshold [0.999, 1.001])",
])

if not overall_pass:
    sys.exit(1)
