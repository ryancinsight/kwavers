#!/usr/bin/env python3
"""
Timing diagnostic for sd_focussed_detector_3D.

Records pressure at 3 locations:
  1. Source point  (42, 31, 31) — test source injection timing
  2. Mid-point     (26, 31, 31) — test propagation mid-way
  3. Bowl apex     (10, 31, 31) — test full propagation to sensor

Compares cross-correlation lags at each point to isolate whether
the timing issue is in source injection vs propagation.
"""

import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_parity_utils import bootstrap_example_paths, DEFAULT_OUTPUT_DIR
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
from kwave.utils.filters import filter_time_series

# ─── Grid/medium constants ───────────────────────────────────────────────────
N = 64
GRID_SIZE = Vector([N, N, N])
DX = 100e-3 / N                     # 1.5625e-3 m
DX_VEC = Vector([DX, DX, DX])
C0   = 1500.0
RHO0 = 1000.0
SOURCE_FREQ = 0.25e6
SOURCE_MAG  = 1.0
PML_SIZE = 10

# Source point (0-indexed)
SX, SY, SZ = 42, 31, 31

# Diagnostic sensor points (0-indexed)
POINTS = {
    "source":  (42, 31, 31),   # At the source
    "midpoint": (26, 31, 31),  # Halfway between source and bowl apex
    "apex":    (10, 31, 31),   # Bowl apex
}

CACHE_DIR = DEFAULT_OUTPUT_DIR
CACHE_KW  = CACHE_DIR / "timing_diag_kwave.npz"
CACHE_PKW = CACHE_DIR / "timing_diag_pykwavers.npz"


def build_config():
    kgrid  = kWaveGrid(GRID_SIZE, DX_VEC)
    medium = kWaveMedium(sound_speed=C0)
    kgrid.makeTime(medium.sound_speed)

    raw = SOURCE_MAG * np.sin(2*np.pi*SOURCE_FREQ * kgrid.t_array)
    sig = filter_time_series(kgrid, medium, raw)
    signal_1d = np.asarray(sig, dtype=np.float64).flatten()

    Nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    print(f"Grid: {N}^3  dx={DX*1e3:.4f}mm  dt={dt:.3e}s  Nt={Nt}")
    print(f"CW period = {1.0/(SOURCE_FREQ*dt):.2f} steps")
    return kgrid, medium, signal_1d, dt, Nt


def run_kwave(kgrid, medium, signal_1d, force=False):
    if CACHE_KW.exists() and not force:
        print("  [k-wave] Loading from cache...")
        d = np.load(CACHE_KW)
        return {k: d[k] for k in d}

    # Sensor mask: the 3 probe points
    sensor_mask = np.zeros((N, N, N), dtype=np.float64)
    for name, (ix, iy, iz) in POINTS.items():
        sensor_mask[ix, iy, iz] = 1.0

    src_mask = np.zeros((N, N, N), dtype=np.float64)
    src_mask[SX, SY, SZ] = 1.0

    source = kSource()
    source.p_mask = src_mask
    source.p = signal_1d.reshape(1, -1)

    sensor = kSensor(sensor_mask.astype(np.int32))

    sim_opts  = SimulationOptions(pml_size=PML_SIZE, data_cast="single", save_to_disk=True)
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

    print("  [k-wave] Running...")
    import time; t0 = time.perf_counter()
    result = kspaceFirstOrder3D(
        medium=medium, kgrid=kgrid,
        source=source, sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [k-wave] Done in {elapsed:.1f} s")

    # sensor_data["p"] shape: either (Nt, n_pts) or (n_pts, Nt)
    arr = np.asarray(result["p"], dtype=np.float64)
    Nt = kgrid.Nt
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == Nt:
        arr = arr.T   # now (n_pts, Nt)

    out = {"elapsed": elapsed}
    names = list(POINTS.keys())
    for i, name in enumerate(names):
        out[f"trace_{name}"] = arr[i] if i < arr.shape[0] else arr[0]
    np.savez(CACHE_KW, **out)
    return out


def run_pykwavers(signal_1d, dt, Nt, force=False):
    if CACHE_PKW.exists() and not force:
        print("  [pykwavers] Loading from cache...")
        d = np.load(CACHE_PKW)
        return {k: d[k] for k in d}

    sensor_mask = np.zeros((N, N, N), dtype=bool)
    for ix, iy, iz in POINTS.values():
        sensor_mask[ix, iy, iz] = True

    src_mask = np.zeros((N, N, N), dtype=np.float64)
    src_mask[SX, SY, SZ] = 1.0

    grid   = pkw.Grid(N, N, N, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_mask(src_mask, signal_1d, SOURCE_FREQ, mode="additive")
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print("  [pykwavers] Running...")
    import time; t0 = time.perf_counter()
    result = sim.run(time_steps=Nt, dt=dt)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    # sensor_data shape: (n_pts, Nt)
    sd = np.asarray(result.sensor_data, dtype=np.float64)
    print(f"  sensor_data shape: {sd.shape}")

    # Map sensor indices to point names
    # The sensor records in the order the mask is flattened (row-major)
    names = list(POINTS.keys())
    pts = list(POINTS.values())

    # Build flat index for each point
    flat_indices = [ix*N*N + iy*N + iz for (ix, iy, iz) in pts]

    # Build full flat sensor mask to find indices
    all_sensor_flat = np.where(sensor_mask.flatten())[0]
    # Map point flat index -> row in sensor_data
    out = {"elapsed": elapsed}
    for i, (name, fidx) in enumerate(zip(names, flat_indices)):
        row_idx = np.where(all_sensor_flat == fidx)[0]
        if len(row_idx) > 0:
            row = row_idx[0]
            out[f"trace_{name}"] = sd[row] if row < sd.shape[0] else sd[0]
        else:
            print(f"  WARNING: Could not find {name} in sensor data")
            out[f"trace_{name}"] = np.zeros(Nt)

    np.savez(CACHE_PKW, **out)
    return out


def cross_corr_lag(a, b, max_lag=20):
    """Return (lag, max_r) where lag maximizes correlation of a and b."""
    nlag = len(a)
    lags = np.arange(-max_lag, max_lag+1)
    corrs = []
    for lag in lags:
        if lag >= 0:
            aa, bb = a[:nlag-lag], b[lag:]
        else:
            aa, bb = a[-lag:], b[:nlag+lag]
        r = np.corrcoef(aa, bb)[0, 1]
        corrs.append(r)
    best_idx = np.argmax(corrs)
    return lags[best_idx], corrs[best_idx]


def analyze(kw, pkw_res, Nt, dt):
    period_steps = 1.0 / (SOURCE_FREQ * dt)
    print(f"\n  CW period = {period_steps:.2f} steps")
    print(f"\n  {'Point':<12} {'Lag (steps)':>12} {'Peak r':>10} {'Note'}")
    print("  " + "-"*60)

    for name in POINTS.keys():
        kw_t  = kw.get(f"trace_{name}", np.zeros(Nt))
        pkw_t = pkw_res.get(f"trace_{name}", np.zeros(Nt))

        # Steady-state: last 50 steps
        kw_ss  = kw_t[-50:]
        pkw_ss = pkw_t[-50:]

        lag, r = cross_corr_lag(kw_ss, pkw_ss, max_lag=int(period_steps)+2)

        # r at lag=0
        r0 = np.corrcoef(kw_ss, pkw_ss)[0, 1]

        note = ""
        if abs(lag) < 2:
            note = "ALIGNED"
        elif abs(abs(lag) - period_steps) < 2:
            note = "~1 period"
        else:
            note = f"{lag/period_steps:.2f} periods"

        print(f"  {name:<12} {lag:>12}  {r:>9.4f}   {note}  (r0={r0:.4f})")


def plot_traces(kw, pkw_res, Nt, dt):
    fig, axes = plt.subplots(len(POINTS), 1, figsize=(12, 3*len(POINTS)), tight_layout=True)
    t_us = np.arange(Nt) * dt * 1e6  # µs

    for ax, (name, (ix, iy, iz)) in zip(axes, POINTS.items()):
        kw_t  = kw.get(f"trace_{name}", np.zeros(Nt))
        pkw_t = pkw_res.get(f"trace_{name}", np.zeros(Nt))

        ax.plot(t_us, kw_t,  label="k-wave (ref)", lw=0.8)
        ax.plot(t_us, pkw_t, label="pykwavers",    lw=0.8, ls="--")
        ax.set_title(f"Sensor: {name} ({ix},{iy},{iz})")
        ax.set_xlabel("t [µs]")
        ax.set_ylabel("p [Pa]")
        ax.legend(fontsize=7)

    out_path = DEFAULT_OUTPUT_DIR / "sd_timing_diag.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  Saved: {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    force = args.no_cache

    print("=== sd_focussed_detector_3D timing diagnostic ===")
    kgrid, medium, signal_1d, dt, Nt = build_config()

    print("\n[1/2] k-wave-python")
    kw = run_kwave(kgrid, medium, signal_1d, force=force)

    print("\n[2/2] pykwavers")
    pkw_res = run_pykwavers(signal_1d, dt, Nt, force=force)

    print("\n=== Timing Analysis ===")
    analyze(kw, pkw_res, Nt, dt)

    plot_traces(kw, pkw_res, Nt, dt)


if __name__ == "__main__":
    main()
