#!/usr/bin/env python3
"""
at_focused_annular_array_3D_full_compare.py
============================================
Full upstream-faithful annular-array forward-sim compare.

Drives the three rings with *independent* CW signals exactly as the
upstream example — 0.5 / 1.0 / 0.75 MPa at 0° / 10° / 20° — so the
focus structure reflects the per-element amplitude/phase prescription.

This exercises the `Source.from_kwave_array_per_element` binding plus
`KWaveArray::build_per_element_source`, which expand an
`[n_elements, n_times]` driving matrix into a per-active-cell signal
`s_cell[c, t] = Σ_i W_i[c] · s_i[t]` using each element's BLI weighted
mask.

Upstream reference:
    external/k-wave-python/examples/at_focused_annular_array_3D/at_focused_annular_array_3D.py

Outputs:
    output/at_focused_annular_array_3D_full_compare.png
    output/at_focused_annular_array_3D_full_metrics.txt

Usage:
    python examples/at_focused_annular_array_3D_full_compare.py [--no-cache] [--allow-failure]
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
from kwave.utils.filters import extract_amp_phase
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.math import round_even
from kwave.utils.signals import create_cw_signals

# Upstream parameters
C0 = 1500.0
RHO0 = 1000.0
SOURCE_F0 = 1.0e6
SOURCE_ROC = 30e-3
DIAMETERS = [
    [0.0, 5.0e-3],
    [10.0e-3, 15.0e-3],
    [20.0e-3, 25.0e-3],
]
SOURCE_AMPS = np.array([0.5e6, 1.0e6, 0.75e6], dtype=np.float64)      # per-ring [Pa]
SOURCE_PHASES_DEG = np.array([0.0, 10.0, 20.0], dtype=np.float64)     # per-ring

AXIAL_SIZE = 40e-3
LATERAL_SIZE = 45e-3
SOURCE_X_OFFSET = 20
PPW = 3
CFL = 0.5

DX = C0 / (PPW * SOURCE_F0)
NX = round_even(AXIAL_SIZE / DX) + SOURCE_X_OFFSET
NY = round_even(LATERAL_SIZE / DX)
NZ = NY

GRID_SIZE = Vector([NX, NY, NZ])
DX_VEC = Vector([DX, DX, DX])

PPP = int(round(PPW / CFL))
DT = 1.0 / (PPP * SOURCE_F0)
T_END = 40e-6
NT = int(round(T_END / DT))

PML_SIZE = 10

SENSOR_IX_LO = SOURCE_X_OFFSET + 2
SENSOR_IX_HI = NX - PML_SIZE - 2
SENSOR_IY = NY // 2
SENSOR_IZ = NZ // 2
N_SENSOR_PTS = SENSOR_IX_HI - SENSOR_IX_LO + 1

HALF_CELL_OFFSET_X = (NX - 1) / 2.0 * DX
HALF_CELL_OFFSET_Y = (NY - 1) / 2.0 * DX
HALF_CELL_OFFSET_Z = (NZ - 1) / 2.0 * DX

APEX_X_KWAVE = (SOURCE_X_OFFSET - (NX - 1) / 2.0) * DX
APEX_X_WORLD = APEX_X_KWAVE + HALF_CELL_OFFSET_X
COC_X_WORLD = APEX_X_WORLD + SOURCE_ROC

KWAVE_APEX = [APEX_X_KWAVE, 0.0, 0.0]
KWAVE_FOCUS_POS = [float(-KWAVE_APEX[0]), 0.0, 0.0]

PARITY_THRESHOLDS = {
    "pearson_r": 0.85,
    "rms_ratio_min": 0.70,
    "rms_ratio_max": 1.35,
    "psnr_db": 14.0,
}

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_full_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_full_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_full_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_full_pykwavers_cache.npz"
CACHE_VERSION = 2


def _load_cached_result(path: Path) -> dict[str, np.ndarray | float] | None:
    if not path.exists():
        return None
    cached = np.load(path, allow_pickle=False)
    version = int(np.asarray(cached["cache_version"]).reshape(())) if "cache_version" in cached.files else 0
    if version != CACHE_VERSION:
        return None
    return {
        "amp_axial": np.asarray(cached["amp_axial"], dtype=np.float64),
        "runtime_s": float(cached["runtime_s"]),
    }


def build_per_element_signals() -> np.ndarray:
    t = np.arange(NT) * DT
    phases_rad = np.deg2rad(SOURCE_PHASES_DEG)
    sig = create_cw_signals(t, SOURCE_F0, SOURCE_AMPS, phases_rad)
    sig = np.asarray(sig, dtype=np.float64)
    if sig.shape != (len(DIAMETERS), NT):
        raise RuntimeError(f"unexpected signal shape {sig.shape}, expected ({len(DIAMETERS)}, {NT})")
    return sig


def run_kwave(per_elem_signals: np.ndarray) -> dict:
    cached = _load_cached_result(_KWAVE_CACHE)
    if cached is not None:
        print("  [k-wave] Loading from cache...")
        return cached

    kgrid = kWaveGrid(GRID_SIZE, DX_VEC)
    kgrid.setTime(NT, DT)
    medium = kWaveMedium(sound_speed=C0, density=RHO0)

    karray = KWaveArray_Kwave(bli_tolerance=0.05, upsampling_rate=10, single_precision=True)
    karray.add_annular_array(KWAVE_APEX, SOURCE_ROC, DIAMETERS, KWAVE_FOCUS_POS)

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, per_elem_signals)

    sensor = kSensor()
    sensor.mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor.mask[SENSOR_IX_LO : SENSOR_IX_HI + 1, SENSOR_IY, SENSOR_IZ] = True
    sensor.record = ["p"]

    sim_opts = SimulationOptions(pml_size=PML_SIZE, pml_inside=True, data_cast="single", save_to_disk=True)
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

    print("  [k-wave] Running 3D PSTD (per-element drive)...")
    t0 = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium, kgrid=kgrid, source=source, sensor=sensor,
        simulation_options=sim_opts, execution_options=exec_opts,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [k-wave] Done in {elapsed:.1f} s")

    p = np.asarray(sensor_data["p"], dtype=np.float64)
    if p.shape[0] == NT:
        p = p.T
    amp, _, _ = extract_amp_phase(p, 1.0 / DT, SOURCE_F0, dim=1, fft_padding=1, window="Rectangular")
    amp_axial = np.asarray(amp, dtype=np.float64).flatten()

    result = {"amp_axial": amp_axial, "runtime_s": elapsed}
    np.savez(_KWAVE_CACHE, cache_version=CACHE_VERSION, **result)
    return result


def run_pykwavers(per_elem_signals: np.ndarray) -> dict:
    cached = _load_cached_result(_PKWAV_CACHE)
    if cached is not None:
        print("  [pykwavers] Loading from cache...")
        return cached

    grid = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    arr = pkw.KWaveArray()
    arr.set_sound_speed(C0)
    arr.set_frequency(SOURCE_F0)
    diameters = [(float(inner), float(outer)) for inner, outer in DIAMETERS]
    bowl_pos = (COC_X_WORLD, HALF_CELL_OFFSET_Y, HALF_CELL_OFFSET_Z)
    arr.add_annular_array(bowl_pos, SOURCE_ROC, diameters)

    source = pkw.Source.from_kwave_array_per_element(
        arr, per_elem_signals, SOURCE_F0, mode="additive"
    )

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[SENSOR_IX_LO : SENSOR_IX_HI + 1, SENSOR_IY, SENSOR_IZ] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print("  [pykwavers] Running CPU PSTD (per-element drive)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    p = np.asarray(result.sensor_data, dtype=np.float64)
    amp, _, _ = extract_amp_phase(p, 1.0 / DT, SOURCE_F0, dim=1, fft_padding=1, window="Rectangular")
    amp_axial = np.asarray(amp, dtype=np.float64).flatten()

    output = {"amp_axial": amp_axial, "runtime_s": elapsed}
    np.savez(_PKWAV_CACHE, cache_version=CACHE_VERSION, **output)
    return output


def plot_comparison(kw: dict, pkw_res: dict) -> None:
    ix_arr = np.arange(SENSOR_IX_LO, SENSOR_IX_HI + 1)
    x_axial_mm = (ix_arr * DX - APEX_X_WORLD) * 1e3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axial_mm, kw["amp_axial"] * 1e-6, "k-", linewidth=1.8, alpha=0.85, label="k-wave-python")
    ax.plot(x_axial_mm, pkw_res["amp_axial"] * 1e-6, "r--", linewidth=1.4, alpha=0.85, label="pykwavers")
    ax.axvline(SOURCE_ROC * 1e3, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
               label=f"geometric focus ({SOURCE_ROC*1e3:.0f} mm)")
    ax.set_xlabel("Axial distance from apex [mm]")
    ax.set_ylabel("Steady-state pressure amplitude [MPa]")
    amps_mpa = (SOURCE_AMPS * 1e-6).tolist()
    ax.set_title(
        "at_focused_annular_array_3D (per-element drive): k-wave vs pykwavers\n"
        f"rings {[(d[0]*1e3, d[1]*1e3) for d in DIAMETERS]} mm  "
        f"|  amps {amps_mpa} MPa  phases {SOURCE_PHASES_DEG.tolist()} deg"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="at_focused_annular_array_3D per-element parity compare.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    if args.no_cache:
        for p in [_KWAVE_CACHE, _PKWAV_CACHE]:
            if p.exists():
                p.unlink()
                print(f"  Removed cache: {p}")

    print("=" * 60)
    print("at_focused_annular_array_3D (per-element drive)")
    print(f"  Grid     : {NX}x{NY}x{NZ}  dx={DX*1e6:.2f} um")
    print(f"  Source   : 3 rings @ {(SOURCE_AMPS*1e-6).tolist()} MPa, {SOURCE_PHASES_DEG.tolist()} deg")
    print(f"  Time     : dt={DT*1e9:.1f} ns  Nt={NT}")
    print("=" * 60)

    per_elem_signals = build_per_element_signals()
    kw = run_kwave(per_elem_signals)
    pkw_res = run_pykwavers(per_elem_signals)

    kw_amp = np.asarray(kw["amp_axial"], dtype=np.float64)
    py_amp = np.asarray(pkw_res["amp_axial"], dtype=np.float64)
    if kw_amp.shape != py_amp.shape:
        raise AssertionError(f"shape mismatch: {kw_amp.shape} != {py_amp.shape}")

    metrics = compute_image_metrics(kw_amp, py_amp)
    print(f"  k-wave peak    = {kw_amp.max()*1e-6:.3f} MPa")
    print(f"  pykwavers peak = {py_amp.max()*1e-6:.3f} MPa")
    print("  --- parity metrics ---")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    thr = PARITY_THRESHOLDS
    checks = {
        f"pearson_r >= {thr['pearson_r']:.2f}": metrics["pearson_r"] >= thr["pearson_r"],
        f"rms_ratio in [{thr['rms_ratio_min']:.2f}, {thr['rms_ratio_max']:.2f}]":
            thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        f"psnr_db >= {thr['psnr_db']:.1f}": metrics["psnr_db"] >= thr["psnr_db"],
    }
    all_pass = all(checks.values())
    status = "PASS" if all_pass else "FAIL"
    for name, ok in checks.items():
        print(f"    {'[OK]' if ok else '[X]'} {name}")
    print(f"  => {status}")

    plot_comparison(kw, pkw_res)

    lines = [
        f"  grid: {NX} x {NY} x {NZ} dx={DX*1e3:.4f} mm",
        f"  rings (inner, outer) [mm]: {[(d[0]*1e3, d[1]*1e3) for d in DIAMETERS]}",
        f"  amplitudes [MPa]: {(SOURCE_AMPS*1e-6).tolist()}",
        f"  phases [deg]:     {SOURCE_PHASES_DEG.tolist()}",
        f"  runtimes: kwave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s",
        "",
        "axial-amplitude parity",
        "----------------------",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"status: {status}")
    lines.append(f"image:  {FIGURE_PATH.name}")
    save_text_report(METRICS_PATH, "at_focused_annular_array_3D_full_compare", lines)

    if all_pass or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
