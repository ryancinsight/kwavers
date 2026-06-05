#!/usr/bin/env python3
"""
ewp_shear_wave_snells_law_compare.py
=====================================
Validates Snell's law for SH shear waves at a planar elastic interface.

Strategy
--------
An SH plane-wave packet (uz polarization, propagating in the x-y plane at
incident angle θ_i from the x-axis) hits a vertical interface (normal along
x) separating two elastic layers with shear speeds c_s1 and c_s2.

Snell's law:
    sin(θ_i) / c_s1 = sin(θ_t) / c_s2

Key: SH waves (uz polarization, wave in x-y plane) undergo NO mode conversion
at a planar interface. Only SH→SH reflection and transmission occur.

Setup
-----
  Grid:      NX=60, NY=60, NZ=1, DX=DY=DZ=1.0 mm
  Layer 1    (x < INTERFACE_X):  c_p=3000, c_s=1500, ρ=1000 kg/m³
  Layer 2    (x ≥ INTERFACE_X):  c_p=5000, c_s=2500, ρ=1000 kg/m³
  PML:       10 cells, pml_inside=True  → valid domain i,j ∈ [10,49]
  Interface: x = 30 mm (i = 30)
  Source:    IVP — initial uz Gaussian plane wave at angle θ_i = 20° from x-axis
             placed in layer 1, centre at (i=18, j=30)
  Sensors:   5 sensors in layer 2 at i=44, j ∈ {34,38,42,46,50}

Measurement
-----------
  Peak arrival time t_peak[s] extracted from vz sensor traces.
  Δt between adjacent sensors → apparent y-phase velocity:
      v_y = Δy / Δt = c_s2 / sin(θ_t_meas)
  → sin(θ_t_meas) = c_s2 × Δt / Δy

Analytical prediction
---------------------
  sin(θ_t_anal) = (c_s2/c_s1) × sin(θ_i)
  With c_s1=1500, c_s2=2500, θ_i=20°:
      sin(θ_t_anal) = (2500/1500) × sin(20°) = 1.667 × 0.342 = 0.570
      θ_t_anal = 34.8°   (<  θ_critical=arcsin(1500/2500)=36.9° ✓)

Parity criterion
----------------
  |θ_t_meas − θ_t_anal| ≤ 1.5°  (FD discretisation + peak-detection noise)

Outputs
-------
  output/ewp_shear_wave_snells_law_compare.png
  output/ewp_shear_wave_snells_law_metrics.txt
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
NX, NY, NZ = 60, 60, 1
DX = 1.0e-3        # [m]

CP1, CS1, RHO1 = 3000.0, 1500.0, 1000.0   # layer 1 (left)
CP2, CS2, RHO2 = 5000.0, 2500.0, 1000.0   # layer 2 (right)
# Stability: c_p² > 2·c_s² (thermodynamic constraint)
assert CP1**2 > 2 * CS1**2
assert CP2**2 > 2 * CS2**2

PML       = 10
INTERFACE_IX = 30              # interface at x = 30 mm
THETA_I_DEG  = 20.0
THETA_I      = np.deg2rad(THETA_I_DEG)

# Analytical Snell's law prediction
sin_theta_t_anal = (CS2 / CS1) * np.sin(THETA_I)
assert sin_theta_t_anal < 1.0, "incident angle exceeds critical angle"
THETA_T_DEG_ANAL = np.rad2deg(np.arcsin(sin_theta_t_anal))

# Time-stepping: CFL based on fastest speed c_p2
DT = 0.25 * DX / CP2           # 5.0e-8 s
T_END = 22.0e-6                 # 22 µs (wave reaches sensors at ~15 µs)
NT = int(T_END / DT)

# IVP pulse: Gaussian plane-wave packet centred at (I0, J0)
I0      = 18              # source centre x-cell (in layer 1, >10+σ from PML)
J0      = 30              # source centre y-cell (grid mid-plane)
SIGMA_CELLS = 3.5         # Gaussian half-width [cells]
AMP     = 1.0             # initial displacement amplitude [m]

# Sensor row in layer 2. Placed ~10 cells past the interface (x=30): far
# enough to clear the interface near-field, close enough that pseudospectral
# propagation dispersion has not yet biased the apparent angle. A dense row of
# sensors, all strictly inside the non-PML interior (valid j ∈ [PML, NY-PML)),
# gives a robust slope fit; the previous sparse row included j=50 which sits in
# the PML and corrupted the timing.
SENSOR_IX = 40                          # x = 40 mm, ~10 cells into layer 2
SENSOR_J  = list(range(22, 45))         # dense in-domain row (j = 22 … 44)
TOL_DEG   = 1.5                         # acceptance threshold [degrees]

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "ewp_shear_wave_snells_law_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ewp_shear_wave_snells_law_metrics.txt"


def build_medium() -> pkw.Medium:
    ix_3d = np.arange(NX)[:, np.newaxis, np.newaxis] * np.ones((1, NY, NZ))
    cp_arr  = np.where(ix_3d < INTERFACE_IX, CP1, CP2)
    cs_arr  = np.where(ix_3d < INTERFACE_IX, CS1, CS2)
    rho_arr = np.where(ix_3d < INTERFACE_IX, RHO1, RHO2)
    return pkw.Medium.elastic_heterogeneous(cp_arr, cs_arr, rho_arr, 1.0)


def build_source() -> pkw.Source:
    """SH plane-wave IVP: uz Gaussian at angle θ_i in x-y plane."""
    ix_2d = np.arange(NX)[:, np.newaxis] * np.ones((1, NY))
    jy_2d = np.arange(NY)[np.newaxis, :] * np.ones((NX, 1))
    # Phase coordinate along propagation direction k̂ = (cos θ, sin θ)
    xi_grid   = ix_2d * np.cos(THETA_I) + jy_2d * np.sin(THETA_I)
    xi_center = I0 * np.cos(THETA_I) + J0 * np.sin(THETA_I)
    uz0_2d = AMP * np.exp(-((xi_grid - xi_center) / SIGMA_CELLS) ** 2)
    # Restrict pulse to layer 1 only (avoids interface artefacts in initial field)
    uz0_2d[INTERFACE_IX:, :] = 0.0
    uz0_3d = uz0_2d[:, :, np.newaxis]  # (NX, NY, 1)
    return pkw.Source.from_initial_displacement(uz0_3d, axis="z")


def run_sim() -> np.ndarray:
    """Returns sensor vz traces (n_sensors, NT)."""
    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = build_medium()
    source = build_source()

    # Sensor mask: 5 points in layer 2 at fixed x, varying y
    sens_mask = np.zeros((NX, NY, NZ), dtype=bool)
    for j in SENSOR_J:
        if 0 <= j < NY:
            sens_mask[SENSOR_IX, j, 0] = True
    sensor = pkw.Sensor.from_mask(sens_mask)

    sim = pkw.Simulation(grid, medium, source, sensor,
                         solver=pkw.SolverType.ElasticPSTD)
    sim.set_pml_size(PML)
    sim.set_pml_inside(True)

    result = sim.run(time_steps=NT, dt=DT)
    vz = result.uz
    if vz is not None:
        data = np.asarray(vz, dtype=np.float64)  # (n_sensors, NT)
    else:
        data = np.asarray(result.sensor_data, dtype=np.float64)
    return data


def first_break_time(trace: np.ndarray, frac: float = 0.2) -> float:
    """First-break (onset) time: the sub-sample instant when ``|trace|`` first
    crosses ``frac`` of its maximum.

    This is the standard refraction-arrival pick. The global peak is unreliable
    here because the displacement-IVP launches a counter-propagating wave whose
    reflected/coda energy can exceed the transmitted wavefront peak at a sensor;
    the onset of the *first* arrival isolates the transmitted wavefront.
    """
    a = np.abs(trace)
    amax = a.max()
    if amax <= 0.0:
        return 0.0
    thr = frac * amax
    idx = int(np.argmax(a > thr))
    if idx <= 0:
        return idx * DT
    y0, y1 = a[idx - 1], a[idx]
    sub = 0.0 if y1 == y0 else (thr - y0) / (y1 - y0)
    return (idx - 1 + sub) * DT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print(
        f"ewp_shear_wave_snells_law:\n"
        f"  c_s1={CS1:.0f}  c_s2={CS2:.0f}  θ_i={THETA_I_DEG:.1f}°  "
        f"θ_t_anal={THETA_T_DEG_ANAL:.2f}°  (θ_critical={np.rad2deg(np.arcsin(CS1/CS2)):.1f}°)\n"
        f"  Grid {NX}×{NY}×{NZ}, dx={DX*1e3:.1f} mm, PML={PML}, "
        f"NT={NT}, dt={DT:.2e} s"
    )

    t0 = time.perf_counter()
    data = run_sim()
    elapsed = time.perf_counter() - t0
    print(f"  Simulation done in {elapsed:.1f} s   sensor_data shape={data.shape}")

    # Sensor ordering: Fortran order (k outermost, i innermost)
    # All sensors at k=0, i=SENSOR_IX → sorted by j ascending
    sensor_j_sorted = sorted(SENSOR_J)
    n_sensors = len(sensor_j_sorted)

    if data.shape[0] != n_sensors:
        print(f"ERROR: expected {n_sensors} sensor rows, got {data.shape[0]}")
        if not args.allow_failure:
            return 1

    # First-break (onset) arrival times of the transmitted wavefront.
    t_peak = np.array([first_break_time(data[s, :]) for s in range(n_sensors)])
    if not np.any(np.abs(data) > 0.0):
        print("ERROR: no wave energy at sensors (insufficient amplitude?)")
        if not args.allow_failure:
            return 1

    # sin(θ_t) from the slope of a robust linear fit of arrival-time vs y:
    #   t(y) = y · sin(θ_t) / c_s2 + const  ⇒  sin(θ_t) = c_s2 · (dt/dy)
    # The least-squares slope over the dense sensor row averages out per-trace
    # onset-pick jitter (far more robust than per-pair finite differences).
    j_arr_fit = np.array(sensor_j_sorted, dtype=float)
    slope_s_per_cell = float(np.polyfit(j_arr_fit, t_peak, 1)[0])
    sin_theta_t_meas = float(np.clip(CS2 * slope_s_per_cell / DX, 0.0, 1.0))
    theta_t_meas_deg = float(np.rad2deg(np.arcsin(np.clip(sin_theta_t_meas, 0, 1))))
    theta_err_deg    = abs(theta_t_meas_deg - THETA_T_DEG_ANAL)

    passed = theta_err_deg <= TOL_DEG
    status = "PASS" if passed else "FAIL"

    print(f"\n  Snell's law [{status}]:")
    print(f"  Status    : {status}")
    print(f"    sin(θ_t) analytical = {sin_theta_t_anal:.4f}   θ_t = {THETA_T_DEG_ANAL:.2f}°")
    print(f"    sin(θ_t) measured   = {sin_theta_t_meas:.4f}   θ_t = {theta_t_meas_deg:.2f}°")
    print(f"    Angular error       = {theta_err_deg:.3f}°  (tolerance ≤ {TOL_DEG}°)")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    t_vec = np.arange(NT) * DT * 1e6  # µs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"SH Wave Snell's Law: c_s1={CS1:.0f} m/s → c_s2={CS2:.0f} m/s  [{status}]\n"
        f"θ_i={THETA_I_DEG:.0f}°  θ_t_anal={THETA_T_DEG_ANAL:.1f}°  "
        f"θ_t_meas={theta_t_meas_deg:.1f}°  err={theta_err_deg:.2f}°",
        fontsize=9,
    )

    colors = plt.cm.viridis(np.linspace(0, 1, n_sensors))
    ax = axes[0]
    for s, j in enumerate(sensor_j_sorted):
        ax.plot(t_vec, data[s, :], color=colors[s], lw=0.9, label=f"j={j}")
        ax.axvline(t_peak[s] * 1e6, color=colors[s], lw=0.6, ls="--")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("vz [m/s]")
    ax.set_title(f"Sensor traces (x=i{SENSOR_IX}×{DX*1e3:.0f}mm, vz component)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ax.scatter(sensor_j_sorted, t_peak * 1e6, c=colors, s=50, zorder=5, label="measured peaks")
    # Fit line through peaks
    j_arr = np.array(sensor_j_sorted)
    t_arr = t_peak
    coeffs = np.polyfit(j_arr, t_arr, 1)
    j_line = np.linspace(j_arr[0], j_arr[-1], 100)
    ax.plot(j_line, np.polyval(coeffs, j_line) * 1e6, "k--", lw=1.0, label="linear fit")
    ax.set_xlabel("Sensor y-cell (j)")
    ax.set_ylabel("Peak arrival time [µs]")
    slope_sin = CS2 * coeffs[0]  # sin(θ_t) from slope
    ax.set_title(
        f"Peak arrival vs y (slope → sin(θ_t)={slope_sin:.3f})\n"
        f"Analytical sin(θ_t)={sin_theta_t_anal:.3f}  Δ={theta_err_deg:.2f}°"
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURE_PATH}")

    save_text_report(METRICS_PATH, "ewp_shear_wave_snells_law_compare", [
        f"Status:              {status}",
        f"theta_i_deg:         {THETA_I_DEG:.4f}",
        f"theta_t_analytical:  {THETA_T_DEG_ANAL:.4f} deg",
        f"sin_theta_t_anal:    {sin_theta_t_anal:.6f}",
        f"sin_theta_t_meas:    {sin_theta_t_meas:.6f}",
        f"theta_t_meas_deg:    {theta_t_meas_deg:.4f} deg",
        f"angular_error_deg:   {theta_err_deg:.4f}  (tolerance <= {TOL_DEG})",
        f"c_s1:                {CS1} m/s",
        f"c_s2:                {CS2} m/s",
        f"runtime_s:           {elapsed:.2f}",
    ])

    if not passed:
        if not args.allow_failure:
            sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
