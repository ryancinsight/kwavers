#!/usr/bin/env python3
"""
at_focused_bowl_3D_compare.py
==============================
Side-by-side comparison of k-wave-python vs pykwavers for a 3D focused
bowl transducer, using the `KWaveArray` off-grid array source abstraction
on both sides.

This is the first pykwavers parity script that exercises:
    pkw.KWaveArray + pkw.KWaveArray.add_bowl_element + pkw.Source.from_kwave_array
which rasterises a spherical-cap bowl onto the 3D grid and drives it with
a shared CW signal.

Key API differences vs k-wave-python:
    * k-wave kWaveArray supports band-limited interpolation (BLI) with
      `bli_tolerance` / `upsampling_rate` — it distributes each off-grid
      element across nearby grid cells with fractional weights.
    * pykwavers uses the same canonical spiral sampling and local BLI stencil
      for the bowl source weights, and the example reports the physical-interior
      source-mass parity separately from the on-axis waveform parity.
    * The waveform parity targets remain conservative because the sensor line is
      still compared against the full steady-state reconstruction path.

Physical setup (matches k-wave-python `at_focused_bowl_3D.py`, except the
sensor is reduced to an on-axis line to keep the run tractable on CPU):
    Grid:      120 x 80 x 80 (dx = 500 um = lambda/3 at 1 MHz)
    Medium:    homogeneous, c0 = 1500 m/s, rho = 1000 kg/m^3, lossless
    Source:    bowl, roc = 30 mm, diameter = 30 mm, apex at x = 10 mm
               facing +X; drive = 1 MHz CW, 1 MPa peak (linear regime)
    Sensor:    on-axis line, ix in [25, 115], iy = iz = 40 (domain centre)
    Time:      dt = 1/(6*1 MHz) ~= 167 ns, Nt = 240 (40 us total)
    PML:       10 pts (inside, matched between both engines)

Outputs:
    output/at_focused_bowl_3D_compare.png     — axial amplitude profile
    output/at_focused_bowl_3D_metrics.txt     — parity metrics

Usage:
    python examples/at_focused_bowl_3D_compare.py
    python examples/at_focused_bowl_3D_compare.py --no-cache
    python examples/at_focused_bowl_3D_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import sys
import time

# Ensure UTF-8 output on Windows (cp1252 terminals reject Unicode subscripts/arrows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    clip_volume_to_physical_interior,
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    compute_trace_metrics,
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
from kwave.utils.filters import extract_amp_phase
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.signals import create_cw_signals

# ---------------------------------------------------------------------------
# Grid / medium / source constants
# ---------------------------------------------------------------------------
# Grid — matches at_focused_bowl_3D.py except Nx is trimmed so the sensor line
# stays short enough to run on CPU in a few minutes.
NX = 120
NY = 80
NZ = 80
GRID_SIZE = Vector([NX, NY, NZ])

C0   = 1500.0        # sound speed [m/s]
RHO0 = 1000.0        # density [kg/m^3]

SOURCE_F0        = 1.0e6      # [Hz]
SOURCE_ROC       = 30e-3      # bowl radius of curvature [m]
SOURCE_DIAMETER  = 30e-3      # bowl aperture diameter [m]
SOURCE_AMP       = 1.0e6      # drive pressure [Pa]  (linear regime)
SOURCE_X_OFFSET  = 20         # grid points from origin to bowl apex
PPW              = 3          # points per wavelength
CFL              = 0.5
DX               = C0 / (PPW * SOURCE_F0)     # 5.0e-4 m
DX_VEC           = Vector([DX, DX, DX])

# Time discretisation — integer points per period for clean FFT amplitude extraction
PPP = int(round(PPW / CFL))                   # 6 points per period
DT  = 1.0 / (PPP * SOURCE_F0)                 # ~1.667e-7 s
T_END = 40e-6
NT  = int(round(T_END / DT))                  # 240

PML_SIZE = 10

# ---------------------------------------------------------------------------
# Bowl geometry mapping
# ---------------------------------------------------------------------------
# pykwavers world coordinates: origin at [0,0,0], grid spans [0, Nx*dx] x ...
# k-wave centered coordinates: x_vec spans ~[-Nx*dx/2, Nx*dx/2], centred on 0.
# Offset: pykwavers_coord = kwave_coord + [Nx*dx/2, Ny*dy/2, Nz*dz/2]
# (approximately; half-cell offset from k-wave's cell-centre convention ignored)
PKW_CENTER_Y   = NY * DX / 2.0                 # lateral centreline [m]
PKW_CENTER_Z   = NZ * DX / 2.0
PKW_APEX_X     = SOURCE_X_OFFSET * DX          # bowl apex [m] in pykwavers world
PKW_COC_X      = PKW_APEX_X + SOURCE_ROC       # centre of curvature [m]
# pykwavers rasterise_bowl uses `center` = centre of curvature, bowl surface
# is the spherical cap facing ±X within the aperture cylinder — only the -X
# face is inside the grid, giving a bowl facing +X with apex at PKW_APEX_X.

# k-wave bowl: give apex and focus_pos such that the axis is +X and the
# centre of curvature lands at kwave-coord equivalent of PKW_COC.
KWAVE_APEX = [-NX * DX / 2.0 + SOURCE_X_OFFSET * DX, 0.0, 0.0]   # kwave-centred
KWAVE_FOCUS_POS = [KWAVE_APEX[0] + SOURCE_ROC, 0.0, 0.0]          # +roc along +X

# ---------------------------------------------------------------------------
# Sensor (on-axis line)
# ---------------------------------------------------------------------------
SENSOR_IX_LO = 25    # start well past the bowl surface
SENSOR_IX_HI = 115   # stop just inside the PML
SENSOR_IY = NY // 2
SENSOR_IZ = NZ // 2
N_SENSOR_PTS = SENSOR_IX_HI - SENSOR_IX_LO + 1

# ---------------------------------------------------------------------------
# Parity targets for the on-axis waveform comparison.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS = {
    "pearson_r":    0.90,
    "rms_ratio_min": 0.75,
    "rms_ratio_max": 1.30,
    "psnr_db":      14.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "at_focused_bowl_3D_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "at_focused_bowl_3D_metrics.txt"

_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "at_focused_bowl_3D_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "at_focused_bowl_3D_pykwavers_cache.npz"


# ---------------------------------------------------------------------------
# Build shared CW drive signal (computed once, used by both engines)
# ---------------------------------------------------------------------------
def build_signal() -> np.ndarray:
    """Return a (NT,) CW sine at SOURCE_F0 with amplitude SOURCE_AMP."""
    t = np.arange(NT) * DT
    sig = create_cw_signals(t, SOURCE_F0, np.array([SOURCE_AMP]), np.array([0.0]))
    return np.asarray(sig, dtype=np.float64).flatten()


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(signal_1d: np.ndarray) -> dict:
    if _KWAVE_CACHE.exists():
        print("  [k-wave] Loading from cache...")
        d = np.load(_KWAVE_CACHE)
        kgrid = kWaveGrid(GRID_SIZE, DX_VEC)
        karray = KWaveArray_Kwave(
            bli_tolerance=0.01, upsampling_rate=10, single_precision=True
        )
        karray.add_bowl_element(KWAVE_APEX, SOURCE_ROC, SOURCE_DIAMETER, KWAVE_FOCUS_POS)
        return {
            "amp_axial": d["amp_axial"],
            "runtime_s": float(d["runtime_s"]),
            "source_weights": np.asarray(karray.get_array_grid_weights(kgrid), dtype=np.float64),
        }

    kgrid = kWaveGrid(GRID_SIZE, DX_VEC)
    kgrid.setTime(NT, DT)
    medium = kWaveMedium(sound_speed=C0, density=RHO0)

    karray = KWaveArray_Kwave(
        bli_tolerance=0.01, upsampling_rate=10, single_precision=True
    )
    karray.add_bowl_element(KWAVE_APEX, SOURCE_ROC, SOURCE_DIAMETER, KWAVE_FOCUS_POS)
    source_weights = np.asarray(karray.get_array_grid_weights(kgrid), dtype=np.float64)

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, signal_1d.reshape(1, -1))

    sensor = kSensor()
    sensor.mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor.mask[SENSOR_IX_LO : SENSOR_IX_HI + 1, SENSOR_IY, SENSOR_IZ] = True
    sensor.record = ["p"]

    sim_opts = SimulationOptions(
        pml_size=PML_SIZE, pml_inside=True,
        data_cast="single", save_to_disk=True,
    )
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

    print("  [k-wave] Running 3D PSTD...")
    t0 = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium, kgrid=kgrid, source=source, sensor=sensor,
        simulation_options=sim_opts, execution_options=exec_opts,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [k-wave] Done in {elapsed:.1f} s")

    # Extract steady-state amplitude at SOURCE_F0 for each sensor point.
    p = np.asarray(sensor_data["p"], dtype=np.float64)
    # p shape is either (Nt, n_sensor) or (n_sensor, Nt); normalise to (n_sensor, Nt).
    if p.shape[0] == NT:
        p = p.T
    amp, _, _ = extract_amp_phase(
        p, 1.0 / DT, SOURCE_F0, dim=1, fft_padding=1, window="Rectangular"
    )
    amp_axial = np.asarray(amp, dtype=np.float64).flatten()
    if amp_axial.size != N_SENSOR_PTS:
        raise RuntimeError(
            f"k-wave amp shape {amp_axial.shape} != expected ({N_SENSOR_PTS},)"
        )

    result = {"amp_axial": amp_axial, "runtime_s": elapsed, "source_weights": source_weights}
    np.savez(_KWAVE_CACHE, **result)
    return result


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(signal_1d: np.ndarray) -> dict:
    if _PKWAV_CACHE.exists():
        print("  [pykwavers] Loading from cache...")
        d = np.load(_PKWAV_CACHE)
        grid = pkw.Grid(NX, NY, NZ, DX, DX, DX)
        arr = pkw.KWaveArray()
        arr.set_sound_speed(C0)
        arr.set_frequency(SOURCE_F0)
        arr.add_bowl_element((PKW_COC_X, PKW_CENTER_Y, PKW_CENTER_Z), SOURCE_ROC, SOURCE_DIAMETER)
        return {
            "amp_axial": d["amp_axial"],
            "runtime_s": float(d["runtime_s"]),
            "source_weights": np.asarray(arr.get_array_weighted_mask(grid), dtype=np.float64),
        }

    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    arr = pkw.KWaveArray()
    arr.set_sound_speed(C0)
    arr.set_frequency(SOURCE_F0)
    # pykwavers add_bowl_element: `position` = centre of curvature.
    arr.add_bowl_element((PKW_COC_X, PKW_CENTER_Y, PKW_CENTER_Z),
                         SOURCE_ROC, SOURCE_DIAMETER)
    source_weights = np.asarray(arr.get_array_weighted_mask(grid), dtype=np.float64)

    source = pkw.Source.from_kwave_array(arr, signal_1d, SOURCE_F0, mode="additive")

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[SENSOR_IX_LO : SENSOR_IX_HI + 1, SENSOR_IY, SENSOR_IZ] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print("  [pykwavers] Running CPU PSTD...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    # result.sensor_data: (n_sensor, Nt)
    p = np.asarray(result.sensor_data, dtype=np.float64)
    amp, _, _ = extract_amp_phase(
        p, 1.0 / DT, SOURCE_F0, dim=1, fft_padding=1, window="Rectangular"
    )
    amp_axial = np.asarray(amp, dtype=np.float64).flatten()

    output = {"amp_axial": amp_axial, "runtime_s": elapsed, "source_weights": source_weights}
    np.savez(_PKWAV_CACHE, **output)
    return output


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def run_comparison() -> dict:
    """Run the bowl comparison and return the raw results plus metrics."""
    signal_1d = build_signal()
    kw = run_kwave(signal_1d)
    pkw_res = run_pykwavers(signal_1d)

    kw_amp = np.asarray(kw["amp_axial"], dtype=np.float64)
    py_amp = np.asarray(pkw_res["amp_axial"], dtype=np.float64)
    if kw_amp.shape != py_amp.shape:
        raise AssertionError(f"on-axis amplitude shape mismatch: {kw_amp.shape} != {py_amp.shape}")

    kw_source_weights = clip_volume_to_physical_interior(
        np.asarray(kw["source_weights"], dtype=np.float64),
        (PML_SIZE, PML_SIZE, PML_SIZE),
    )
    py_source_weights = clip_volume_to_physical_interior(
        np.asarray(pkw_res["source_weights"], dtype=np.float64),
        (PML_SIZE, PML_SIZE, PML_SIZE),
    )
    if kw_source_weights.shape != py_source_weights.shape:
        raise AssertionError(
            f"source weight shape mismatch: {kw_source_weights.shape} != {py_source_weights.shape}"
        )

    metrics = compute_image_metrics(kw_amp, py_amp)
    source_metrics = compute_trace_metrics(kw_source_weights, py_source_weights)

    return {
        "kwave": kw,
        "pykwavers": pkw_res,
        "summary": metrics,
        "source_metrics": source_metrics,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict) -> None:
    # Axial position [mm] relative to bowl apex
    ix_arr = np.arange(SENSOR_IX_LO, SENSOR_IX_HI + 1)
    x_axial_mm = (ix_arr * DX - PKW_APEX_X) * 1e3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axial_mm, kw["amp_axial"] * 1e-6,  "k-",
            linewidth=1.8, alpha=0.85, label="k-wave-python (BLI)")
    ax.plot(x_axial_mm, pkw_res["amp_axial"] * 1e-6, "r--",
            linewidth=1.4, alpha=0.85, label="pykwavers (native KWaveArray)")
    ax.axvline(SOURCE_ROC * 1e3, color="gray", linestyle=":",
               linewidth=1.0, alpha=0.6,
               label=f"geometric focus ({SOURCE_ROC*1e3:.0f} mm from apex)")
    ax.set_xlabel("Axial distance from bowl apex [mm]")
    ax.set_ylabel("Steady-state pressure amplitude [MPa]")
    ax.set_title(
        "at_focused_bowl_3D: k-wave-python vs pykwavers\n"
        f"bowl: roc={SOURCE_ROC*1e3:.0f} mm, diameter={SOURCE_DIAMETER*1e3:.0f} mm  "
        f"|  f0={SOURCE_F0*1e-6:.1f} MHz, {SOURCE_AMP*1e-6:.1f} MPa CW  "
        f"|  grid {NX}x{NY}x{NZ}, dx={DX*1e6:.0f} um"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for at_focused_bowl_3D."
    )
    parser.add_argument("--no-cache", action="store_true",
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
    print("at_focused_bowl_3D: k-wave-python vs pykwavers")
    print(f"  Grid     : {NX}x{NY}x{NZ}  dx={DX*1e6:.2f} um  "
          f"(domain {NX*DX*1e3:.1f} x {NY*DX*1e3:.1f} x {NZ*DX*1e3:.1f} mm)")
    print(f"  Medium   : c0={C0} m/s  rho={RHO0} kg/m^3  (lossless)")
    print(f"  Source   : CW bowl  roc={SOURCE_ROC*1e3:.1f} mm  "
          f"dia={SOURCE_DIAMETER*1e3:.1f} mm  amp={SOURCE_AMP*1e-6:.1f} MPa  "
          f"f0={SOURCE_F0*1e-6:.1f} MHz")
    print(f"  Apex     : x={PKW_APEX_X*1e3:.2f} mm  (pykwavers world)")
    print(f"  CoC      : x={PKW_COC_X*1e3:.2f} mm")
    print(f"  Sensor   : on-axis line ix in [{SENSOR_IX_LO}, {SENSOR_IX_HI}] "
          f"(n={N_SENSOR_PTS})  at iy=iz={SENSOR_IY}")
    print(f"  Time     : dt={DT*1e9:.1f} ns  Nt={NT}  t_end={T_END*1e6:.0f} us")
    print(f"  PML      : {PML_SIZE} pts (inside)")
    print("=" * 60)

    print("\n[1/2] Running comparison...")
    result = run_comparison()
    kw = result["kwave"]
    pkw_res = result["pykwavers"]
    metrics = result["summary"]
    source_metrics = result["source_metrics"]

    print(f"  k-wave peak amp on axis     = {kw['amp_axial'].max()*1e-6:.3f} MPa")
    print(f"  pykwavers peak amp on axis = {pkw_res['amp_axial'].max()*1e-6:.3f} MPa")

    print("\n--- Parity evaluation ---")
    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r":  metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio":  thr["rms_ratio_min"]  <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":    metrics["psnr_db"]    >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status: {status}")
    print(f"  Pearson r = {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  RMS ratio = {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  PSNR      = {metrics['psnr_db']:.2f} dB  "
          f"(target >= {thr['psnr_db']} dB)  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  Source Pearson r = {source_metrics['pearson_r']:.6f}")
    print(f"  Source RMS ratio = {source_metrics['rms_ratio']:.6f}")
    print(f"  Source RMSE      = {source_metrics['rmse']:.6e}")
    print(f"  Source peak ratio= {source_metrics['peak_ratio']:.6f}")

    plot_comparison(kw, pkw_res)

    header = "\n".join([
        "at_focused_bowl_3D parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}x{NZ}   dx={DX:.6e} m",
        f"source: bowl  roc={SOURCE_ROC} m  dia={SOURCE_DIAMETER} m  "
        f"amp={SOURCE_AMP} Pa  f0={SOURCE_F0} Hz",
        f"apex_kwavers_world: x={PKW_APEX_X:.6e} m  (center iy={SENSOR_IY}, iz={SENSOR_IZ})",
        f"dt={DT:.6e} s   Nt={NT}",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "note: source weights are compared on the physical interior after clipping the PML halo",
        "",
    ])
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f}  (target >= {thr['psnr_db']} dB)",
        "",
        "source weights (physical interior): k-wave-python vs pykwavers",
        f"  pearson_r = {source_metrics['pearson_r']:.6f}",
        f"  rms_ratio = {source_metrics['rms_ratio']:.6f}",
        f"  rmse      = {source_metrics['rmse']:.6e}",
        f"  peak_ratio= {source_metrics['peak_ratio']:.6f}",
        f"peak_amp_kwave_Pa      = {float(kw['amp_axial'].max()):.6e}",
        f"peak_amp_pykwavers_Pa  = {float(pkw_res['amp_axial'].max()):.6e}",
        f"peak_ratio             = "
        f"{float(pkw_res['amp_axial'].max())/(float(kw['amp_axial'].max())+1e-30):.6f}",
    ]
    save_text_report(METRICS_PATH, header, report_lines)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if status == "FAIL" and not args.allow_failure:
        raise SystemExit(
            "at_focused_bowl_3D parity targets not met. "
            "Run with --allow-failure to collect diagnostics."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
