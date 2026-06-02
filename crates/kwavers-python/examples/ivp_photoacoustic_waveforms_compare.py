#!/usr/bin/env python3
"""
ivp_photoacoustic_waveforms_compare.py
=======================================
Side-by-side comparison of k-wave-python vs pykwavers for the canonical
initial-value-problem (IVP) photoacoustic waveform example.

A small ball of elevated initial pressure is placed at the centre of a
64×64×64 grid, and the resulting acoustic waveform is recorded at a single
sensor point 10 grid spacings to the right of centre.  This directly
exercises the `Source.from_initial_pressure(p0)` API in pykwavers (the
equivalent of `source.p0` in k-Wave / k-wave-python).

Physical setup (matches k-wave-python `ivp_photoacoustic_waveforms.py` 3D leg):
  Grid:    64×64×64,  dx = 1 mm / 64 = 15.625 µm  (1 mm cubic domain)
  Medium:  homogeneous, c₀ = 1500 m/s, ρ = 1000 kg/m³, lossless
  Source:  initial pressure ball, radius = 2 grid pts, centred at [32,32,32]
           (1-indexed) = [31,31,31] (0-indexed);  amplitude = 1 Pa
  Sensor:  single point at [Nx//2 + 10, Nx//2, Nx//2] = [42, 32, 32] (0-indexed)
           records pressure waveform
  Time:    dt = 2 ns,  Nt = 150  (t_end = 300 ns)
  PML:     10 grid points (default k-wave)

Output:
  output/ivp_photoacoustic_waveforms_compare.png  — trace overlay
  output/ivp_photoacoustic_waveforms_metrics.txt  — parity metrics

Usage:
  python examples/ivp_photoacoustic_waveforms_compare.py
  python examples/ivp_photoacoustic_waveforms_compare.py --no-cache
  python examples/ivp_photoacoustic_waveforms_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.mapgen import make_ball

# ---------------------------------------------------------------------------
# Grid / medium / source constants (match k-wave-python example exactly)
# ---------------------------------------------------------------------------
NX = NY = NZ = 64
GRID_SIZE = Vector([NX, NY, NZ])
X = 1e-3                            # domain size [m] = 1 mm
DX = X / NX                         # 15.625 µm

C0   = 1500.0                       # sound speed [m/s]
RHO0 = 1000.0                       # density [kg/m³]

SOURCE_RADIUS          = 2          # [grid points]  — ball radius
SOURCE_SENSOR_DISTANCE = 10         # [grid points]  — centre-to-sensor offset
SOURCE_AMP             = 1.0        # initial pressure amplitude [Pa]

# Sensor location (0-indexed NumPy)
# k-wave example: sensor.mask[Nx//2 + source_sensor_distance, Nx//2, Nx//2] = True
SENSOR_IX = NX // 2 + SOURCE_SENSOR_DISTANCE   # 42
SENSOR_IY = NY // 2                            # 32
SENSOR_IZ = NZ // 2                            # 32

# Ball source centre (1-indexed, for make_ball)
BALL_CENTER = Vector([NX // 2, NY // 2, NZ // 2])   # [32, 32, 32]

DT    = 2e-9    # time step [s]
T_END = 300e-9  # simulation end time [s]
NT    = int(round(T_END / DT))      # 150

PML_SIZE = 10

# ---------------------------------------------------------------------------
# Parity targets (IVP single-trace comparison; tighter than CW due to
# deterministic wavefront — only numerical dispersion separates the two)
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS = {
    "pearson_r":    0.97,
    "rms_ratio_min": 0.85,
    "rms_ratio_max": 1.15,
    "psnr_db":      22.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "ivp_photoacoustic_waveforms_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ivp_photoacoustic_waveforms_metrics.txt"

_KWAVE_CACHE  = DEFAULT_OUTPUT_DIR / "ivp_photoacoustic_kwave_cache.npz"
_PKWAV_CACHE  = DEFAULT_OUTPUT_DIR / "ivp_photoacoustic_pykwavers_cache.npz"


# ---------------------------------------------------------------------------
# Step 1 — Shared initial pressure field (identical for both engines)
# ---------------------------------------------------------------------------
def build_p0() -> np.ndarray:
    """
    Create the 3D initial pressure distribution.

    Returns a (NX, NY, NZ) float64 array with SOURCE_AMP Pa inside the ball
    and 0 Pa outside.  make_ball uses 1-indexed grid positions.
    """
    ball_mask = make_ball(GRID_SIZE, BALL_CENTER, SOURCE_RADIUS)
    p0 = SOURCE_AMP * np.asarray(ball_mask, dtype=np.float64)
    n_pts = int(p0.sum())
    vol   = n_pts * DX**3 * 1e12   # µL
    print(f"  Initial pressure ball: {n_pts} grid pts  "
          f"(≈{n_pts * (4/3 * np.pi) ** (1/3) * DX * 1e6:.1f} µm effective radius)")
    return p0


# ---------------------------------------------------------------------------
# Step 2 — k-wave-python (3D, CPU)
# ---------------------------------------------------------------------------
def run_kwave(p0: np.ndarray) -> dict:
    """Run k-wave-python with source.p0 and a single sensor point."""
    if _KWAVE_CACHE.exists():
        print("  [k-wave] Loading from cache...")
        d = np.load(_KWAVE_CACHE)
        return {"trace": d["trace"], "runtime_s": float(d["runtime_s"])}

    kgrid  = kWaveGrid(GRID_SIZE, Vector([DX, DX, DX]))
    kgrid.setTime(NT, DT)
    medium = kWaveMedium(sound_speed=C0)

    source       = kSource()
    source.p0    = p0                       # (NX, NY, NZ) initial pressure [Pa]

    sensor_mask  = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[SENSOR_IX, SENSOR_IY, SENSOR_IZ] = True
    sensor       = kSensor(sensor_mask)
    sensor.record = ["p"]

    sim_opts  = SimulationOptions(pml_size=PML_SIZE, data_cast="single", save_to_disk=True)
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

    print("  [k-wave] Running 3D PSTD...")
    t0 = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium, kgrid=kgrid,
        source=source, sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [k-wave] Done in {elapsed:.1f} s")

    # sensor_data["p"]: (1, NT) or (NT,) depending on k-wave version
    trace = np.asarray(sensor_data["p"], dtype=np.float64).flatten()  # → (NT,)
    result = {"trace": trace, "runtime_s": elapsed}
    np.savez(_KWAVE_CACHE, **result)
    return result


# ---------------------------------------------------------------------------
# Step 3 — pykwavers CPU PSTD
# ---------------------------------------------------------------------------
def run_pykwavers(p0: np.ndarray) -> dict:
    """
    Run pykwavers with Source.from_initial_pressure(p0).

    Sensor: single-point mask at [SENSOR_IX, SENSOR_IY, SENSOR_IZ].
    result.sensor_data shape: (1, NT) → flatten → (NT,).
    """
    if _PKWAV_CACHE.exists():
        print("  [pykwavers] Loading from cache...")
        d = np.load(_PKWAV_CACHE)
        return {"trace": d["trace"], "runtime_s": float(d["runtime_s"])}

    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    # k-wave-python's `kspaceFirstOrder3D` defaults to smooth_p0=True, which
    # applies a Blackman window to p0 before injection. pykwavers' source
    # injection is unsmoothed, so we replicate k-wave's smoothing here so the
    # two engines see the same effective initial pressure.
    from kwave.utils.filters import smooth as kwave_smooth
    p0_smoothed = np.asarray(kwave_smooth(p0, restore_max=True), dtype=np.float64)
    source = pkw.Source.from_initial_pressure(p0_smoothed)

    # `Sensor.from_mask` expects a 3-D boolean ndarray.
    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[SENSOR_IX, SENSOR_IY, SENSOR_IZ] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print("  [pykwavers] Running CPU PSTD...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    # result.sensor_data: (1, NT) → flatten → (NT,)
    trace = np.asarray(result.sensor_data, dtype=np.float64).flatten()
    output = {"trace": trace, "runtime_s": elapsed}
    np.savez(_PKWAV_CACHE, **output)
    return output


# ---------------------------------------------------------------------------
# Step 4 — Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict) -> None:
    """Single panel overlay: k-wave (black solid) vs pykwavers (red dashed)."""
    t_ns   = np.arange(NT) * DT * 1e9   # time axis [ns]
    kw_tr  = kw["trace"]
    pkw_tr = pkw_res["trace"]

    # Expected wave arrival: centre is at [31,31,31] (0-idx), sensor at [42,32,32]
    # propagation distance ≈ sqrt(11²+1²+1²)*DX ≈ 11.1*15.625 µm ≈ 173 µm
    # arrival time ≈ 173e-6 / 1500 ≈ 115 ns
    expected_arrival_ns = np.sqrt((SENSOR_IX - NX//2 + 1)**2 +
                                   (SENSOR_IY - NY//2)**2 +
                                   (SENSOR_IZ - NZ//2)**2) * DX / C0 * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Full trace
    ax = axes[0]
    ax.plot(t_ns, kw_tr,  "k-",  linewidth=1.8, alpha=0.85, label="k-wave-python")
    ax.plot(t_ns, pkw_tr, "r--", linewidth=1.4, alpha=0.85, label="pykwavers")
    ax.axvline(expected_arrival_ns, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
               label=f"expected arrival ≈{expected_arrival_ns:.0f} ns")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_title("Photoacoustic waveform — full trace")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Zoom around the peak
    peak_idx = int(np.argmax(np.abs(kw_tr)))
    half_win  = max(5, NT // 10)
    i0 = max(0, peak_idx - half_win)
    i1 = min(NT, peak_idx + half_win)
    ax2 = axes[1]
    ax2.plot(t_ns[i0:i1], kw_tr[i0:i1],  "k-",  linewidth=1.8, alpha=0.85, label="k-wave-python")
    ax2.plot(t_ns[i0:i1], pkw_tr[i0:i1], "r--", linewidth=1.4, alpha=0.85, label="pykwavers")
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("Pressure [Pa]")
    ax2.set_title("Photoacoustic waveform — wavefront zoom")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "ivp_photoacoustic_waveforms_3D: k-wave-python vs pykwavers\n"
        f"Ball source (radius={SOURCE_RADIUS} pts) · sensor at {SOURCE_SENSOR_DISTANCE} pts offset"
        f" · grid {NX}³ · dx={DX*1e6:.2f} µm · dt={DT*1e9:.0f} ns",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for ivp_photoacoustic_waveforms (3D)."
    )
    parser.add_argument("--no-cache",      action="store_true",
                        help="Delete cached results and force a fresh run.")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail (diagnostics).")
    args = parser.parse_args()

    if args.no_cache:
        for p in [_KWAVE_CACHE, _PKWAV_CACHE]:
            if p.exists():
                p.unlink()
                print(f"  Removed cache: {p}")

    print("=" * 60)
    print("ivp_photoacoustic_waveforms_3D: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}³   dx = {DX*1e6:.4f} µm   (1 mm domain)")
    print(f"  Medium : c₀ = {C0} m/s  lossless  (rho = {RHO0} kg/m³)")
    print(f"  Source : initial pressure ball, radius={SOURCE_RADIUS} pts, {SOURCE_AMP:.0f} Pa")
    print(f"  Sensor : single point at [{SENSOR_IX},{SENSOR_IY},{SENSOR_IZ}] (0-indexed)")
    print(f"  Time   : dt={DT*1e9:.0f} ns,  Nt={NT},  t_end={T_END*1e9:.0f} ns")
    print(f"  PML    : {PML_SIZE} pts (inside)")
    print("=" * 60)

    # --- Build shared initial pressure ---
    print("\n[1/3] Building initial pressure field (make_ball)...")
    p0 = build_p0()

    # --- k-wave run ---
    print("\n[2/3] k-wave-python (3D PSTD)...")
    kw_result = run_kwave(p0)
    kw_tr = kw_result["trace"]
    print(f"  peak = {np.abs(kw_tr).max():.6f} Pa   rms = {np.sqrt(np.mean(kw_tr**2)):.6f} Pa")

    # --- pykwavers run ---
    print("\n[3/3] pykwavers (CPU PSTD, from_initial_pressure)...")
    pkw_result = run_pykwavers(p0)
    pkw_tr = pkw_result["trace"]
    print(f"  peak = {np.abs(pkw_tr).max():.6f} Pa   rms = {np.sqrt(np.mean(pkw_tr**2)):.6f} Pa")

    # --- Parity evaluation ---
    print("\n--- Parity evaluation ---")
    metrics = compute_image_metrics(kw_tr, pkw_tr)
    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r":  metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio":  thr["rms_ratio_min"]  <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":    metrics["psnr_db"]    >= thr["psnr_db"],
    }
    status   = "PASS" if all(checks.values()) else "FAIL"
    all_pass = (status == "PASS")

    print(f"  Status: {status}")
    print(f"  Pearson r = {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  RMS ratio = {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  PSNR      = {metrics['psnr_db']:.2f} dB  "
          f"(target >= {thr['psnr_db']} dB)  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  Peak kwave  = {float(np.abs(kw_tr).max()):.6e} Pa")
    print(f"  Peak pkwav  = {float(np.abs(pkw_tr).max()):.6e} Pa  "
          f"(ratio = {float(np.abs(pkw_tr).max()) / (float(np.abs(kw_tr).max())+1e-30):.4f})")

    # --- Figure ---
    plot_comparison(kw_result, pkw_result)

    # --- Text report ---
    header = "\n".join([
        "ivp_photoacoustic_waveforms_3D parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}x{NZ}   dx={DX:.6e} m",
        f"source: initial pressure ball, radius={SOURCE_RADIUS} pts, amp={SOURCE_AMP} Pa",
        f"sensor: single point at [{SENSOR_IX},{SENSOR_IY},{SENSOR_IZ}] (0-indexed)",
        f"dt={DT:.6e} s   Nt={NT}   t_end={T_END:.6e} s",
        f"kwave_runtime_s: {kw_result['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_result['runtime_s']:.3f}",
        "",
    ])
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f}  (target >= {thr['psnr_db']} dB)",
        f"peak_kwave_Pa     = {float(np.abs(kw_tr).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(pkw_tr).max()):.6e}",
        f"peak_ratio        = {float(np.abs(pkw_tr).max())/(float(np.abs(kw_tr).max())+1e-30):.6f}",
    ]
    save_text_report(METRICS_PATH, header, report_lines)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if not all_pass and not args.allow_failure:
        raise SystemExit(
            "ivp_photoacoustic_waveforms_3D parity targets not met. "
            "Run with --allow-failure to collect diagnostics."
        )

    print("\nDone.")
    print(f"  Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
