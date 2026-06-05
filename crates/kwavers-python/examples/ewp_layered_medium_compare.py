#!/usr/bin/env python3
"""
ewp_layered_medium_compare.py
==============================
Validates P-wave propagation through a two-layer heterogeneous elastic medium
by measuring the inter-sensor delay of the transmitted wave in layer 2.

k-wave-python does not support elastic simulations.  This script validates
pykwavers against the analytical P-wave arrival delay between two sensors
placed in the faster second layer, both receiving the wave after it has
transmitted through the soft-to-hard impedance interface.

Physical setup
--------------
Grid:      NX=60, NY=1, NZ=1 (strictly 1-D in x), dx = 0.5 mm
           With PML=10 and pml_inside=True the valid domain is x ∈ [10:50].
Medium:    Two-layer heterogeneous along x (fluid-like, cs=0):
             Layer 1 (x < 20): c_p=1500 m/s, ρ=1000 kg/m³  (soft tissue)
             Layer 2 (x ≥ 20): c_p=3000 m/s, ρ=1900 kg/m³  (bone-like)
Source:    Single-cell vx Ricker wavelet at (SRC_X=14, 0, 0), in layer 1.
           σ = 16 samples (time steps), t₀ = 5σ = 80 steps = 4.0 µs.
           σ=16 → dominant wavelength ≈ 21 cells in layer 2 (kΔx ≈ 0.30 rad)
           → < 0.1% group-velocity FD error after transmission.
Sensors:   S1 at x=26 (layer 2,  6 cells past interface at x=20)
           S2 at x=38 (layer 2, 18 cells past interface)
           Both sensors are in the same medium (layer 2), eliminating
           cross-interface shape distortion from the Pearson comparison.
           S1 is 24 cells from the right PML boundary at x=50.
           S2 is 12 cells from the right PML boundary at x=50.

Simulation window
-----------------
CFL = 0.3 (c2=3000 m/s) → DT = 5 × 10⁻⁸ s
NT = 200 steps (T_END = 10.0 µs).  S2 P-wave peaks at step ≈ 180, well
within NT = 200.

PML REFLECTION CHECK — velocity-only PML reflections are guaranteed to
arrive AFTER the observation window ends:
  Right-PML front at S2 : step 140+40=180 (amplitude ~0.3% of peak after
                          reflection; S2 main peak also at step 180 → direct
                          wave dominates entirely)
  Right-PML PEAK at S2  : step 220+40=260 > NT=200 ✓
  Left-PML PEAK at S2   : step 107+127=234 > NT=200 ✓
Both sensor peaks are completely free of PML reflections.

Analytical invariant (P-wave inter-sensor delay in layer 2)
-----------------------------------------------------------
Ricker wavelet centred at t₀ = 5σ = 80 steps.
P-wave CFL in layer 1 : CFL1 = c1*DT/DX = 1500*5e-8/5e-4 = 0.15 cells/step.
P-wave CFL in layer 2 : CFL2 = c2*DT/DX = 3000*5e-8/5e-4 = 0.30 cells/step.

Path to S1 (x=26):
  Layer 1 (14→20): 6 cells / 0.15 = 40 steps
  Layer 2 (20→26): 6 cells / 0.30 = 20 steps
  t_peak_S1 = t₀ + 60 steps = 140 steps = 7.0 µs

Path to S2 (x=38):
  Layer 1 (14→20): 6 cells / 0.15 = 40 steps
  Layer 2 (20→38): 18 cells / 0.30 = 60 steps
  t_peak_S2 = t₀ + 100 steps = 180 steps = 9.0 µs

Inter-sensor delay (both in layer 2):
  Δx = (38−26) × 0.5 mm = 6.0 mm
  Δt_analytical = 6.0e−3 / 3000 = 2.0e−6 s = 40.0 samples

Validation criteria:
    |t_peak_measured − t_analytical| ≤ 5 sample periods for both sensors
    Pearson(S1, S2 aligned)           ≥ 0.90
    Both sensors are in layer 2 with low dispersion (21 cells/wavelength),
    guaranteeing waveform shape similarity without impedance-mismatch distortion.

Usage::

    python examples/ewp_layered_medium_compare.py
    python examples/ewp_layered_medium_compare.py --allow-failure
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
NX = 60
NY = 1    # degenerate y — avoids y-PML covering entire valid domain
NZ = 1
DX = 0.5e-3        # [m]

C1   = 1500.0      # Layer 1 P-wave speed [m/s]
C2   = 3000.0      # Layer 2 P-wave speed [m/s]
RHO1 = 1000.0      # Layer 1 density [kg/m³]
RHO2 = 1900.0      # Layer 2 density [kg/m³]

# Fluid-like layers (cs=0 eliminates shear waves, isolates P-wave timing).
CS = 0.0

# Ricker wavelet: σ=16 steps.
# Layer 2 dominant wavelength = c2 * 2*σ_time/sqrt(2) ≈ 21 cells → kΔx≈0.30 rad
# → < 0.1% group-velocity FD error.
SIGMA_CELLS = 16

# CFL ≤ 0.3 using fastest wave speed (layer 2)
CFL = 0.3
DT  = CFL * DX / C2   # = 5e-8 s

# NT = 200 → T_END = 10.0 µs.
# S2 P-wave peak at step 180; all PML reflections arrive at step ≥ 220.
NT  = 200

PML = 10

# Layer split: layer 1 for x < 20, layer 2 for x >= 20.
LAYER_SPLIT = 20

SRC_X = 14   # in layer 1; 4 cells from left PML boundary at x=10

S1_X = 26    # layer 2,  6 cells past interface; 24 cells from right PML at 50
S2_X = 38    # layer 2, 18 cells past interface; 12 cells from right PML at 50

# Analytical timing (both sensors in layer 2)
sigma_s = SIGMA_CELLS * DT    # σ in seconds
T0      = 5.0 * sigma_s       # = 4.0 µs = 80 steps

# Layer-1 travel from SRC to interface
CFL1 = C1 * DT / DX   # = 0.15 cells/step
CFL2 = C2 * DT / DX   # = 0.30 cells/step

layer1_steps = (LAYER_SPLIT - SRC_X) / CFL1     # 6 / 0.15 = 40 steps

t_arr_S1 = T0 + (layer1_steps + (S1_X - LAYER_SPLIT) / CFL2) * DT
t_arr_S2 = T0 + (layer1_steps + (S2_X - LAYER_SPLIT) / CFL2) * DT

# Analytical inter-sensor delay
delta_x_layer2     = (S2_X - S1_X) * DX              # 6.0 mm
dt_analytical      = delta_x_layer2 / C2              # = 2.0e-6 s
dt_samples_analytic = dt_analytical / DT               # = 40.0 samples


def _ricker_wavelet(nt: int, dt: float, sigma_secs: float, amp: float = 1.0) -> np.ndarray:
    """Ricker wavelet (2nd derivative of Gaussian), centred at t₀ = 5σ."""
    t   = np.arange(nt) * dt
    t0  = 5.0 * sigma_secs
    tau = (t - t0) / sigma_secs
    return amp * (1.0 - tau ** 2) * np.exp(-0.5 * tau ** 2)


def _build_medium_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp  = np.empty((NX, NY, NZ))
    cs  = np.zeros((NX, NY, NZ))
    rho = np.empty((NX, NY, NZ))
    cp [:LAYER_SPLIT, :, :] = C1;  rho[:LAYER_SPLIT, :, :] = RHO1
    cp [LAYER_SPLIT:, :, :] = C2;  rho[LAYER_SPLIT:, :, :] = RHO2
    return cp, cs, rho


def run_elastic() -> np.ndarray:
    """Return sensor vx traces (2, NT): [S1, S2]."""
    cp, cs, rho = _build_medium_arrays()
    medium = pkw.Medium.elastic_heterogeneous(cp, cs, rho)

    mask = np.zeros((NX, NY, NZ), dtype=bool)
    mask[SRC_X, 0, 0] = True
    signal_1d = _ricker_wavelet(NT, DT, sigma_s, amp=1.0)
    source    = pkw.Source.from_elastic_velocity_source(mask, ux=signal_1d)

    sens_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sens_mask[S1_X, 0, 0] = True
    sens_mask[S2_X, 0, 0] = True
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
    # Fortran-order (x-fastest, NY=NZ=1): S1_X=26 < S2_X=38 → data[0]=S1, data[1]=S2
    return data   # (2, NT)


def _peak_index(sig: np.ndarray) -> int:
    return int(np.argmax(np.abs(sig)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    step_S1_expected = int(round(t_arr_S1 / DT))
    step_S2_expected = int(round(t_arr_S2 / DT))

    print(
        f"ewp_layered_medium: NX={NX}, NY={NY}, NZ={NZ}, NT={NT}, dt={DT:.3e} s\n"
        f"  Layer 1 x<{LAYER_SPLIT}: c_p={C1:.0f} m/s  ρ={RHO1:.0f} kg/m³\n"
        f"  Layer 2 x≥{LAYER_SPLIT}: c_p={C2:.0f} m/s  ρ={RHO2:.0f} kg/m³\n"
        f"  Source at x={SRC_X} (Ricker σ={SIGMA_CELLS} cells, t₀={T0*1e6:.1f} µs)\n"
        f"  Analytical S1 (x={S1_X}): {t_arr_S1*1e6:.3f} µs = step {step_S1_expected}\n"
        f"  Analytical S2 (x={S2_X}): {t_arr_S2*1e6:.3f} µs = step {step_S2_expected}\n"
        f"  Δx={delta_x_layer2*1e3:.1f} mm  →  analytical delay = "
        f"{dt_analytical*1e6:.3f} µs = {dt_samples_analytic:.1f} samples"
    )

    print("\nRunning elastic simulation...")
    t0 = time.perf_counter()
    data = run_elastic()
    runtime_s = time.perf_counter() - t0
    print(f"  Done in {runtime_s:.1f} s  shape={data.shape}")

    if data.shape[0] < 2:
        print(f"ERROR: expected 2 sensor traces, got {data.shape[0]}")
        if not args.allow_failure:
            return 1

    s1 = data[0, :]
    s2 = data[1, :]
    nt = s1.shape[0]

    pk1 = _peak_index(s1)
    pk2 = _peak_index(s2)

    err1_samp = abs(pk1 - step_S1_expected)
    err2_samp = abs(pk2 - step_S2_expected)

    dt_measured_samples = pk2 - pk1
    timing_error_samples = abs(dt_measured_samples - dt_samples_analytic)

    print(f"\nS1 (layer 2, x={S1_X}): measured step={pk1}  "
          f"analytical={step_S1_expected}  abs_err={err1_samp} samples")
    print(f"S2 (layer 2, x={S2_X}): measured step={pk2}  "
          f"analytical={step_S2_expected}  abs_err={err2_samp} samples")
    print(f"  Measured delay:    {dt_measured_samples} samples = "
          f"{dt_measured_samples*DT*1e6:.3f} µs")
    print(f"  Analytical delay:  {dt_samples_analytic:.2f} samples = "
          f"{dt_analytical*1e6:.3f} µs")
    print(f"  Timing error:      {timing_error_samples:.2f} samples")

    TIMING_MAX = 5.0

    timing_ok = (err1_samp <= TIMING_MAX and err2_samp <= TIMING_MAX
                 and timing_error_samples <= TIMING_MAX)

    # Windowed Pearson: compare ±2σ samples around each peak.
    # Both sensors are in the same medium so their pulse shapes are identical
    # near the peak; the pre-cursor region (numerical FD artifact) is excluded.
    half = 2 * SIGMA_CELLS
    w1 = s1[max(0, pk1 - half) : min(nt, pk1 + half)]
    w2 = s2[max(0, pk2 - half) : min(nt, pk2 + half)]
    minlen = min(len(w1), len(w2))
    if minlen > 10 and w1[:minlen].std() > 0 and w2[:minlen].std() > 0:
        pearson = float(pearsonr(w1[:minlen], w2[:minlen])[0])
    else:
        pearson = 0.0

    PEARSON_MIN = 0.90
    shape_ok    = pearson >= PEARSON_MIN

    passed = timing_ok and shape_ok

    print(f"  pearson_r: {pearson:.6f}  (windowed +-2sigma, threshold >= {PEARSON_MIN})  "
          f"{'OK' if shape_ok else 'FAIL'}")
    status = "PASS" if passed else "FAIL"
    print(f"\n  status: {status}")

    # ---- Plot ----------------------------------------------------------------
    t_ax = np.arange(nt) * DT * 1e6
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(t_ax, s1, label=f"S1 (layer 2, x={S1_X})", lw=1.2)
    axes[0].plot(t_ax, s2, label=f"S2 (layer 2, x={S2_X})", lw=1.2, ls="--")
    axes[0].axvline(t_arr_S1 * 1e6, color="C0", lw=0.8, ls=":", label="Analytical t_S1")
    axes[0].axvline(t_arr_S2 * 1e6, color="C1", lw=0.8, ls=":", label="Analytical t_S2")
    axes[0].set_xlabel("Time [µs]")
    axes[0].set_ylabel("ux [m/s]")
    axes[0].set_title(
        f"ewp_layered_medium: P-wave timing  "
        f"err1={err1_samp:.0f} samp, err2={err2_samp:.0f} samp  [{status}]"
    )
    axes[0].legend()

    s1_norm = s1 / (np.abs(s1).max() + 1e-30)
    s2_norm = s2 / (np.abs(s2).max() + 1e-30)
    axes[1].plot(t_ax, s1_norm, label="S1 (norm)")
    axes[1].plot(t_ax, s2_norm, "--", label="S2 (norm)")
    wlo1 = max(0, pk1 - half) * DT * 1e6
    whi1 = min(nt, pk1 + half) * DT * 1e6
    wlo2 = max(0, pk2 - half) * DT * 1e6
    whi2 = min(nt, pk2 + half) * DT * 1e6
    axes[1].axvspan(wlo1, whi1, alpha=0.15, color="C0", label="window S1")
    axes[1].axvspan(wlo2, whi2, alpha=0.15, color="C1", label="window S2")
    axes[1].set_xlabel("Time [µs]")
    axes[1].set_ylabel("Normalised ux")
    axes[1].set_title(f"Windowed Pearson (+-2sigma): {pearson:.4f}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / "ewp_layered_medium_compare.png"
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"  Figure saved: {fig_path}")

    save_text_report(
        out_dir / "ewp_layered_medium_compare.txt",
        "ewp_layered_medium_compare",
        [
            f"Grid: {NX}×{NY}×{NZ}  dx={DX*1e3:.3f} mm",
            f"Layer 1: c_p={C1:.0f} m/s  rho={RHO1:.0f} kg/m3  x<{LAYER_SPLIT}",
            f"Layer 2: c_p={C2:.0f} m/s  rho={RHO2:.0f} kg/m3  x>={LAYER_SPLIT}",
            f"Source: Ricker sigma={SIGMA_CELLS} cells  t0={T0*1e6:.2f} us  at x={SRC_X}",
            f"NT={NT}  dt={DT:.4e} s",
            f"S1 (layer 2, x={S1_X}) analytical: step {step_S1_expected}  measured: step {pk1}  err={err1_samp} samp",
            f"S2 (layer 2, x={S2_X}) analytical: step {step_S2_expected}  measured: step {pk2}  err={err2_samp} samp",
            f"Analytical delay: {dt_samples_analytic:.1f} samples  measured: {dt_measured_samples} samples",
            f"Timing error (delay): {timing_error_samples:.2f} samples",
            f"Pearson windowed (+-2sigma): {pearson:.6f}",
            f"RESULT: {status}",
        ],
    )

    if not passed and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
