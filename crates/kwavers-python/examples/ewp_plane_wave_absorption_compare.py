#!/usr/bin/env python3
"""
ewp_plane_wave_absorption_compare.py
======================================
Validates P-wave propagation speed in a homogeneous elastic solid using a
1-D geometry (Ny=Nz=1) to produce a clean compressional plane wave with no
S-wave contamination from transverse boundary effects.

k-wave-python does not support elastic simulations.  This script validates
pykwavers against the analytical P-wave arrival delay between two sensors.

Physical setup
--------------
Grid:      NX=60, Ny=1, Nz=1 (strictly 1-D in x), dx = 0.5 mm
           With PML=10 and pml_inside=True the valid domain is x ∈ [10:50].
Medium:    Homogeneous elastic solid (bone-like)
           c_p = 3000 m/s  (P-wave speed)
           c_s = 1500 m/s  (S-wave speed — inactive in 1-D)
           ρ   = 1900 kg/m³
Source:    Single-cell vx source at (SRC_X=20, 0, 0).
           Ricker wavelet, σ = 20 samples, t₀ = 5σ = 100 steps.
           σ=20 → dominant kΔx ≈ 0.25 rad → < 0.05% phase-velocity FD error.
           The large σ drives spectral content at kΔx=π/2 (the super-dispersive
           short-wavelength band responsible for pre-cursors) to exp(−158) ≈ 0,
           eliminating pre-cursor contamination entirely.
           In 1-D the single cell is equivalent to a plane wave; degenerate
           y and z axes (Ny=Nz=1) yield fd1_y=fd1_z=0, so only P-waves
           propagate in x at speed c_p.
Sensors:   S_near at (SX_NEAR=30, 0, 0) — 10 cells from source.
           S_far  at (SX_FAR=38, 0, 0)  — 18 cells from source.
           Both lie in the valid non-PML domain (< 50).
           S_far=38 is 12 cells from right PML (boundary at 50).

Simulation window
-----------------
T_END = 10.0 µs (NT = 200 steps at dt = 5 × 10⁻⁸ s).
The P-wave peak at S_far arrives at step ≈ 160, well within NT = 200.
PML REFLECTION CHECK — velocity-only PML reflections arrive after NT=200:
  Right-PML peak at S_far  : step 160 + 80 = 240 > NT ✓
    (peak to right PML x=50: 12 cells / 0.3 = 40 steps;
     reflected to x=38: 40 steps; total 80 steps after far peak)
  Left-PML peak at S_near  : step 133 + 134 = 267 > NT ✓
    (peak to left PML x=10: 20 cells / 0.3 = 67 steps;
     reflected to x=30: 67 steps; total 134 steps after near peak)
Both sensors see only the direct wave within NT = 200.

Analytical inter-sensor delay
------------------------------
Δx = (18−10) × 0.5 mm = 4.0 mm
Δt = Δx / c_p = 4.0e−3 / 3000 = 1.333e−6 s = 26.67 samples

Expected peak steps (σ=20 cells, t₀=100 steps, CFL=0.3):
    S_near (x=30): 100 + 10/0.3 ≈ 133
    S_far  (x=38): 100 + 18/0.3 = 160

Validation criteria:
    |Δt_measured − Δt_analytical|  ≤ 2 samples
    Pearson windowed (+-2σ around each peak)  ≥ 0.90
    σ=20 eliminates pre-cursor (source spectral content at kΔx=π/2 → 0),
    so the windowed comparison captures only the clean Ricker pulse shape.
    Dominat kΔx ≈ 0.25 rad → < 0.05% phase-velocity FD error → timing
    error < 0.1 samples for both sensors.

Usage::

    python examples/ewp_plane_wave_absorption_compare.py
    python examples/ewp_plane_wave_absorption_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

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
NX = 60           # x cells (propagation axis)
NY = 1            # degenerate y — no transverse waves, no S-wave contamination
NZ = 1            # degenerate z
DX = 0.5e-3       # [m]

CP  = 3000.0      # P-wave speed [m/s]
CS  = 1500.0      # S-wave speed [m/s] (inactive for 1-D grid)
RHO = 1900.0      # density [kg/m³]

# Ricker wavelet: σ = 20 time-steps, t₀ = 5σ (near-zero start amplitude).
# σ=20 → dominant kΔx ≈ 0.25 rad → < 0.05% phase-velocity FD error.
# Large σ drives spectral content at superdispersive kΔx=π/2 to exp(-158)≈0,
# eliminating the pre-cursor that corrupts timing at close sensors.
SIGMA_CELLS = 20

# CFL ≤ 0.3 (elastic stability uses fastest wave speed)
CFL = 0.3
DT  = CFL * DX / CP   # = 5e-8 s

# NT = 200: captures S_far peak at step 160 + 40-step tail margin.
# All PML reflections arrive at step ≥ 240 > NT = 200.
T_END = 10.0e-6
NT    = int(round(T_END / DT))   # = 200

PML = 10

# Valid (non-PML) domain for pml_inside=True, PML=10, NX=60: cells [10:50].
# Source at SRC_X=20 (10 cells inside left PML boundary at 10).
SRC_X = 20

# Sensor offsets from source (cells)
NEAR_OFFSET = 10   # → SX_NEAR = 30  (20 cells from right PML boundary at 50)
FAR_OFFSET  = 18   # → SX_FAR  = 38  (12 cells from right PML boundary at 50)

SX_NEAR = SRC_X + NEAR_OFFSET   # = 30
SX_FAR  = SRC_X + FAR_OFFSET    # = 38


def _ricker_wavelet(nt: int, dt: float, sigma_s: float, amp: float = 1.0) -> np.ndarray:
    """Ricker wavelet (2nd derivative of Gaussian), centred at t₀ = 5σ."""
    t   = np.arange(nt) * dt
    t0  = 5.0 * sigma_s
    tau = (t - t0) / sigma_s
    return amp * (1.0 - tau ** 2) * np.exp(-0.5 * tau ** 2)


def run_elastic() -> np.ndarray:
    """Return sensor ux data (2, NT) for [S_near, S_far]."""
    cp_arr  = np.full((NX, NY, NZ), CP)
    cs_arr  = np.full((NX, NY, NZ), CS)
    rho_arr = np.full((NX, NY, NZ), RHO)
    medium  = pkw.Medium.elastic_heterogeneous(cp_arr, cs_arr, rho_arr)

    sigma_s   = SIGMA_CELLS * DT
    signal_1d = _ricker_wavelet(NT, DT, sigma_s, amp=1.0)

    # Single-cell source (1-D geometry: Ny=Nz=1).
    src_mask = np.zeros((NX, NY, NZ), dtype=bool)
    src_mask[SRC_X, 0, 0] = True
    source = pkw.Source.from_elastic_velocity_source(src_mask, ux=signal_1d)

    # Single-point sensors.
    sens_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sens_mask[SX_NEAR, 0, 0] = True
    sens_mask[SX_FAR,  0, 0] = True
    sensor = pkw.Sensor.from_mask(sens_mask)

    sim = pkw.Simulation(
        pkw.Grid(NX, NY, NZ, DX, DX, DX),
        medium, source, sensor,
        solver=pkw.SolverType.ElasticPSTD,
    )
    sim.set_pml_size(PML)
    sim.set_pml_inside(True)

    result = sim.run(time_steps=NT, dt=DT)
    ux = result.ux
    if ux is not None:
        data = np.asarray(ux, dtype=np.float64)
    else:
        data = np.asarray(result.sensor_data, dtype=np.float64)
    return data   # shape (2, NT) — [S_near, S_far]


def _peak_index(sig: np.ndarray, expected: int, half: int) -> int:
    """Robust peak detection within [expected-half, expected+half].

    Restricts argmax to a window around the analytically expected arrival,
    preventing the numerical FD pre-cursor from corrupting peak detection.
    """
    lo = max(0, expected - half)
    hi = min(len(sig), expected + half)
    return lo + int(np.argmax(np.abs(sig[lo:hi])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Analytical P-wave delay between S_near and S_far
    delta_x             = (FAR_OFFSET - NEAR_OFFSET) * DX   # 8 × 0.5 mm = 4.0 mm
    dt_analytical       = delta_x / CP                       # [s]
    dt_samples_analytic = dt_analytical / DT                 # = 26.67 samples

    print(
        f"ewp_plane_wave_absorption (1-D): NX={NX}, NY={NY}, NZ={NZ}, "
        f"NT={NT}, dt={DT:.3e} s\n"
        f"  c_p={CP:.0f} m/s, c_s={CS:.0f} m/s, rho={RHO:.0f} kg/m³\n"
        f"  Source at x={SRC_X}, sensors at x={SX_NEAR} and x={SX_FAR}\n"
        f"  Δx={delta_x*1e3:.1f} mm  →  analytical P-wave delay = "
        f"{dt_analytical*1e6:.3f} µs = {dt_samples_analytic:.2f} samples"
    )

    print("\nRunning 1-D elastic simulation...")
    t0 = time.perf_counter()
    data = run_elastic()
    runtime_s = time.perf_counter() - t0
    print(f"  Done in {runtime_s:.1f} s  sensor_data shape={data.shape}")

    if data.shape[0] < 2:
        print("ERROR: expected 2 sensor traces, got", data.shape[0])
        return 1

    # Sensor Fortran-order (x-fastest in SensorRecorder, NY=NZ=1):
    # only x varies → SX_NEAR=30 < SX_FAR=38 → data[0]=S_near, data[1]=S_far.
    sig_near = data[0, :]
    sig_far  = data[1, :]
    nt       = sig_near.shape[0]

    # Analytically expected peak steps: t0 = 5*σ; travel time = offset/(CFL).
    t0_steps   = 5 * SIGMA_CELLS
    pk_near_expected = t0_steps + int(round(NEAR_OFFSET / (CFL)))
    pk_far_expected  = t0_steps + int(round(FAR_OFFSET  / (CFL)))
    # Search window: ±3σ around expected arrival (covers ±50% travel-time error).
    search_half = 3 * SIGMA_CELLS

    pk_near = _peak_index(sig_near, pk_near_expected, search_half)
    pk_far  = _peak_index(sig_far,  pk_far_expected,  search_half)
    dt_measured_samples = pk_far - pk_near
    dt_measured         = dt_measured_samples * DT
    timing_error_samples = abs(dt_measured_samples - dt_samples_analytic)

    print(f"\nP-wave peak indices: near={pk_near}, far={pk_far}")
    print(f"  Measured delay:    {dt_measured_samples} samples = "
          f"{dt_measured*1e6:.3f} µs")
    print(f"  Analytical delay:  {dt_samples_analytic:.2f} samples = "
          f"{dt_analytical*1e6:.3f} µs")
    print(f"  Timing error:      {timing_error_samples:.2f} samples")

    # Windowed Pearson: compare ±2σ samples around each peak.
    # Excludes the numerical pre-cursor region (which arrives well before
    # the physical wave) and evaluates only waveform shape near the pulse.
    half = 2 * SIGMA_CELLS
    w_near = sig_near[max(0, pk_near - half) : min(nt, pk_near + half)]
    w_far  = sig_far [max(0, pk_far  - half) : min(nt, pk_far  + half)]
    minlen = min(len(w_near), len(w_far))
    if minlen > 10 and w_near[:minlen].std() > 0 and w_far[:minlen].std() > 0:
        pearson = float(pearsonr(w_near[:minlen], w_far[:minlen])[0])
    else:
        pearson = 0.0

    print(f"  Pearson windowed (+-2sigma around peaks) = {pearson:.6f}")

    TIMING_MAX_SAMPLES = 2
    PEARSON_MIN        = 0.90

    timing_ok = timing_error_samples <= TIMING_MAX_SAMPLES
    pearson_ok = pearson >= PEARSON_MIN
    passed     = timing_ok and pearson_ok

    print(f"\nMetrics:")
    print(f"  Timing error {timing_error_samples:.2f} samples  "
          f"(threshold ≤ {TIMING_MAX_SAMPLES})  {'OK' if timing_ok else 'FAIL'}")
    print(f"  pearson_r: {pearson:.6f}  (threshold >= {PEARSON_MIN})  "
          f"{'OK' if pearson_ok else 'FAIL'}")
    status = "PASS" if passed else "FAIL"
    print(f"\n  status: {status}")

    # ---- Plot ----------------------------------------------------------------
    t_ax = np.arange(nt) * DT * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t_ax, sig_near, label=f"S_near (x={SX_NEAR})", lw=1.2)
    axes[0].plot(t_ax, sig_far,  label=f"S_far  (x={SX_FAR})",  lw=1.2, ls="--")
    axes[0].axvline(pk_near * DT * 1e6, color="C0", lw=0.8, ls=":", alpha=0.6)
    axes[0].axvline(pk_far  * DT * 1e6, color="C1", lw=0.8, ls=":", alpha=0.6)
    axes[0].set_xlabel("Time [µs]")
    axes[0].set_ylabel("Ux amplitude [m/s]")
    axes[0].set_title(
        f"ewp_plane_wave (1-D): P-wave timing  "
        f"delay={dt_measured_samples} samp (analytical={dt_samples_analytic:.1f}), "
        f"Pearson(win)={pearson:.4f}  [{status}]"
    )
    axes[0].legend()

    # Show windowed region on normalised plot
    near_norm = sig_near / (np.abs(sig_near).max() + 1e-30)
    far_norm  = sig_far  / (np.abs(sig_far ).max() + 1e-30)
    axes[1].plot(t_ax, near_norm, label="S_near (norm)")
    axes[1].plot(t_ax, far_norm,  "--", label="S_far (norm)")
    wlo = max(0, pk_near - half) * DT * 1e6
    whi = min(nt, pk_near + half) * DT * 1e6
    axes[1].axvspan(wlo, whi, alpha=0.15, color="C0", label="window S_near")
    wlo2 = max(0, pk_far - half) * DT * 1e6
    whi2 = min(nt, pk_far  + half) * DT * 1e6
    axes[1].axvspan(wlo2, whi2, alpha=0.15, color="C1", label="window S_far")
    axes[1].set_xlabel("Time [µs]")
    axes[1].set_ylabel("Normalised amplitude")
    axes[1].set_title(f"Windowed Pearson (+-2sigma): {pearson:.4f}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / "ewp_plane_wave_absorption_compare.png"
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"  Figure saved: {fig_path}")

    save_text_report(
        out_dir / "ewp_plane_wave_absorption_compare.txt",
        "ewp_plane_wave_absorption_compare",
        [
            f"Grid: NX={NX}, NY={NY}, NZ={NZ}  dx={DX*1e3:.3f} mm",
            f"c_p={CP:.0f} m/s  c_s={CS:.0f} m/s  rho={RHO:.0f} kg/m3",
            f"Source at x={SRC_X}",
            f"Sensors: S_near x={SX_NEAR}, S_far x={SX_FAR}",
            f"NT={NT}  dt={DT:.4e} s",
            f"Analytical P-wave delay: {dt_samples_analytic:.2f} samples",
            f"Measured P-wave delay: {dt_measured_samples} samples",
            f"Timing error: {timing_error_samples:.2f} samples",
            f"Pearson windowed (+-2sigma): {pearson:.6f}",
            f"RESULT: {status}",
        ],
    )

    if not passed and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
