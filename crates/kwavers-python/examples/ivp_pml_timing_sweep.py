#!/usr/bin/env python3
"""
ivp_pml_timing_sweep.py
Test pykwavers IVP timing with various PML sizes (including 0 = no PML)
to determine if PML causes the 7-step timing offset.

Compare against cached k-Wave results.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[2]
_pyp  = _root / "pykwavers" / "python"
if str(_pyp) not in sys.path:
    sys.path.insert(0, str(_pyp))

import pykwavers as pkw
from kwave.utils.mapgen import make_ball
from kwave.data import Vector
from kwave.utils.filters import smooth as kwave_smooth

OUTPUT_DIR = Path(__file__).parent / "output"
KWAVE_CACHE = OUTPUT_DIR / "ivp_photoacoustic_kwave_cache.npz"

NX = NY = NZ = 64
DX = 1e-3 / NX
C0 = 1500.0
RHO0 = 1000.0
DT = 2e-9
NT = 150
SOURCE_RADIUS = 2
SENSOR_IX, SENSOR_IY, SENSOR_IZ = 42, 32, 32

def pearson_r(a, b):
    am = a - a.mean(); bm = b - b.mean()
    d = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d) if d > 1e-30 else 0.0

def find_zero_crossing(arr):
    """Find first positive-to-negative crossing (earliest wavefront marker)."""
    for i in range(len(arr)-1):
        if arr[i] > 0 and arr[i+1] <= 0:
            return i  # 0-indexed
    return -1

def run_pykwavers_ivp(p0_smooth, pml_size, pml_inside=True):
    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_initial_pressure(p0_smooth.copy())

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask[SENSOR_IX, SENSOR_IY, SENSOR_IZ] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(pml_size)
    if pml_inside:
        sim.set_pml_inside(True)

    result = sim.run(time_steps=NT, dt=DT)
    trace = np.asarray(result.sensor_data, dtype=np.float64).flatten()
    return trace

def main():
    # Build p0_smooth (same as compare script)
    ball_mask = make_ball(Vector([NX,NY,NZ]), Vector([NX//2,NY//2,NZ//2]), SOURCE_RADIUS)
    p0_raw    = np.asarray(ball_mask, dtype=np.float64)
    p0_smooth = kwave_smooth(p0_raw, restore_max=True)
    print(f"p0: {int(p0_raw.sum())} pts, smooth peak={p0_smooth.max():.6f}")

    # Load k-Wave reference
    kw = np.load(KWAVE_CACHE)["trace"].flatten().astype(np.float64)
    t_ns = np.arange(NT) * DT * 1e9

    kw_zc = find_zero_crossing(kw)
    print(f"k-Wave  zero-cross: step {kw_zc+1} (t={t_ns[kw_zc]:.1f} ns)")

    phy_t_ns = np.sqrt((SENSOR_IX-31)**2 + (SENSOR_IY-31)**2 + (SENSOR_IZ-31)**2) * DX / C0 * 1e9
    print(f"Physical arrival:   {phy_t_ns:.1f} ns = step {phy_t_ns/(DT*1e9):.1f}")
    print()

    # Sweep PML sizes
    pml_sizes = [0, 2, 5, 10]
    print("=" * 75)
    print(f"{'PML':>4}  {'zc_step':>8}  {'zc_ns':>8}  {'r_no_shift':>12}  {'best_lag':>10}  {'r_at_lag':>10}")
    print("=" * 75)

    from scipy import signal as sp_signal

    for pml in pml_sizes:
        trace = run_pykwavers_ivp(p0_smooth, pml_size=pml)
        zc = find_zero_crossing(trace)
        zc_str = f"{zc+1}" if zc >= 0 else "N/A"
        zc_ns  = t_ns[zc] if zc >= 0 else float('nan')
        r_raw = pearson_r(kw, trace)

        corr = sp_signal.correlate(kw, trace, mode="full")
        lags = sp_signal.correlation_lags(len(kw), len(trace), mode="full")
        best_lag = int(lags[np.argmax(corr)])

        # r at best lag
        if best_lag >= 0:
            tr_s = np.concatenate([np.zeros(best_lag), trace[:-best_lag if best_lag else None]])
        else:
            tr_s = np.concatenate([trace[-best_lag:], np.zeros(-best_lag)])
        r_lag = pearson_r(kw, tr_s)

        print(f"{pml:4d}  {zc_str:>8}  {zc_ns:8.1f}  {r_raw:12.6f}  {best_lag:10d}  {r_lag:10.6f}")

    print()
    # Also test WITHOUT source_kappa correction: run current pml=10 with and without it
    # (We can't easily disable the Rust-side correction without recompiling,
    # but we can check by passing raw p0 instead of p0_smooth to mimic what k-Wave gets)
    print("--- Testing with raw (unsmoothed) p0 for pykwavers, pml=10 ---")
    trace_raw = run_pykwavers_ivp(p0_raw, pml_size=10)
    zc_raw = find_zero_crossing(trace_raw)
    zc_ns_raw = t_ns[zc_raw] if zc_raw >= 0 else float('nan')
    r_raw_r = pearson_r(kw, trace_raw)
    corr = sp_signal.correlate(kw, trace_raw, mode="full")
    lags = sp_signal.correlation_lags(len(kw), len(trace_raw), mode="full")
    best_raw = int(lags[np.argmax(corr)])
    print(f"  raw p0: zero-cross step {zc_raw+1} ({zc_ns_raw:.1f} ns)  r={r_raw_r:.6f}  best_lag={best_raw}")

    print()
    print("Wavefront comparison (steps 44-65) for pml=0 vs pml=10:")
    trace0  = run_pykwavers_ivp(p0_smooth, pml_size=0)
    trace10 = run_pykwavers_ivp(p0_smooth, pml_size=10)
    print(f"{'Step':>5}  {'kwave':>12}  {'pkw pml=0':>12}  {'pkw pml=10':>12}")
    for i in range(43, 65):
        print(f"{i+1:5d}  {kw[i]:12.4e}  {trace0[i]:12.4e}  {trace10[i]:12.4e}")

if __name__ == "__main__":
    main()
