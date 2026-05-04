#!/usr/bin/env python3
"""
ivp_1D_simulation_compare.py
==============================
Parity comparison for the upstream ``ivp_1D_simulation`` example
(initial-value-problem in a 1-D heterogeneous propagation medium).

Physical setup (matches k-wave-python ``examples/ivp_1D_simulation.py``):
  Grid:    512 points  dx = 0.05 mm
  Medium:  heterogeneous (piecewise constant)
             c[:round(512/3)]  = 2000 m/s   (left third)
             c[round(512/3):]  = 1500 m/s   (right two-thirds)
             rho[:round(4*512/5)-1] = 1000 kg/m³
             rho[round(4*512/5)-1:] = 1500 kg/m³  (rightmost fifth)
  Source:  initial pressure p0 — smoothed sinusoidal pulse
             starting at grid index 280, width 101 points, peak 1.0 Pa
  Sensor:  Cartesian 2-point at x = −10 mm and x = +10 mm
             maps to grid indices 56 and 456 (0-based)
  PML:     default size, inside domain
  t_end:   2.5 × (Nx × dx) / max(c)  =  32 μs

Algorithm
---------
k-wave-python places the 1-D grid with x ∈ [−Nx/2, Nx/2 − 1] × dx so
x = −12.8 mm at index 0 and x = +12.775 mm at index 511.  The Cartesian
sensor points ±10 mm land exactly on grid indices 56 and 456.

Sensor row ordering
-------------------
Both engines produce sensor row 0 = x = −10 mm, row 1 = x = +10 mm.
No C→Fortran permutation is required for the 1-D case because the two
grid-mask True entries are ordered by their linear (x-only) index.

Parity thresholds
-----------------
A 1-D heterogeneous medium with partial reflections and transmission at
impedance boundaries is a demanding test.  Both engines implement the same
k-space pseudospectral scheme so Pearson r ≥ 0.97 and RMS ratio within
[0.85, 1.20] are required.

Outputs
-------
* ``output/ivp_1D_simulation_compare.png``
* ``output/ivp_1D_simulation_metrics.txt``

Usage
-----
  python examples/ivp_1D_simulation_compare.py
  python examples/ivp_1D_simulation_compare.py --no-cache
  python examples/ivp_1D_simulation_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    compute_trace_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical constants  (must match ivp_1D_simulation.py exactly)
# ---------------------------------------------------------------------------
NX = 512
DX = 0.05e-3              # grid spacing [m]

# Sound speed: first third = 2000 m/s, remainder = 1500 m/s
C_FAST = 2000.0           # [m/s]
C_SLOW = 1500.0           # [m/s]
SPLIT_C = round(NX / 3)   # = 171

# Density: last fifth = 1500 kg/m³, rest = 1000 kg/m³
RHO_LOW  = 1000.0         # [kg/m³]
RHO_HIGH = 1500.0         # [kg/m³]
SPLIT_RHO = round(4 * NX / 5) - 1  # 0-based start index of high-rho region = 408

# Sinusoidal pulse geometry
X_POS  = 280              # starting grid index [0-based]
WIDTH  = 100              # pulse spans (width+1) = 101 points
HEIGHT = 1.0              # peak pressure [Pa]

# Sensor positions
# x_min = -(Nx/2)*dx = -256*0.05e-3 = -0.0128 m
# index(x) = (x - x_min) / dx  (0-based)
# index(-10e-3) = (-10e-3 + 12.8e-3) / 0.05e-3 = 56
# index(+10e-3) = ( 10e-3 + 12.8e-3) / 0.05e-3 = 456
SENSOR_X_MM      = [-10.0, 10.0]          # sensor x coordinates [mm]
SENSOR_INDICES   = [56, 456]              # 0-based grid indices
N_SENSORS        = 2

# Simulation duration
T_END = 2.5 * (NX * DX) / C_FAST         # = 32e-6 s = 32 μs

# ---------------------------------------------------------------------------
# Parity thresholds
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r_min":  0.97,
    "rms_ratio_min":  0.85,
    "rms_ratio_max":  1.20,
    "psnr_db_min":    18.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "ivp_1D_simulation_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ivp_1D_simulation_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "ivp_1D_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "ivp_1D_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_cache(path: os.PathLike) -> dict | None:
    if REFRESH_CACHE or not os.path.exists(os.fspath(path)):
        return None
    try:
        d = np.load(os.fspath(path), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "pressure":  np.asarray(d["pressure"], dtype=np.float64),
            "nt":        int(d["nt"]),
            "dt":        float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_cache(
    path: os.PathLike,
    pressure: np.ndarray,
    nt: int,
    dt: float,
    runtime_s: float,
) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        pressure=np.asarray(pressure, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
def _build_c() -> np.ndarray:
    """1-D sound speed array matching ivp_1D_simulation.py."""
    c = np.full(NX, C_SLOW)
    c[:SPLIT_C] = C_FAST
    return c


def _build_rho() -> np.ndarray:
    """1-D density array matching ivp_1D_simulation.py."""
    rho = np.full(NX, RHO_LOW)
    rho[SPLIT_RHO:] = RHO_HIGH
    return rho


def _build_p0_smooth(c: np.ndarray) -> np.ndarray:
    """Sinusoidal pulse p0, then apply k-Wave spatial smoothing."""
    from kwave.utils.filters import smooth as kwave_smooth

    in_arr = np.linspace(0, 2.0 * np.pi, WIDTH + 1)   # 101 points
    pulse  = (HEIGHT / 2.0) * np.sin(in_arr - np.pi / 2.0) + (HEIGHT / 2.0)

    p0 = np.concatenate([
        np.zeros(X_POS),
        pulse,
        np.zeros(NX - X_POS - (WIDTH + 1)),
    ])
    assert p0.size == NX, f"p0 length {p0.size} ≠ {NX}"

    # Apply the same k-space smoothing that kspaceFirstOrder uses by default.
    p0_smooth = np.asarray(kwave_smooth(p0, restore_max=True), dtype=np.float64)
    return p0_smooth


def build_shared_inputs() -> dict:
    """Grid, medium, p0, and nt/dt from k-Wave's makeTime."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid

    kgrid = kWaveGrid(Vector([NX]), Vector([DX]))
    c   = _build_c()
    rho = _build_rho()

    kgrid.makeTime(c, t_end=T_END)
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    p0_smooth = _build_p0_smooth(c)

    return {
        "kgrid":     kgrid,
        "c":         c,
        "rho":       rho,
        "p0_smooth": p0_smooth,
        "nt":        nt,
        "dt":        dt,
    }


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(inputs: dict, *, no_cache: bool = False) -> dict:
    """Run k-wave-python in 1-D with the Cartesian 2-point sensor."""
    if not no_cache:
        cached = _load_cache(_KWAVE_CACHE)
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid   = inputs["kgrid"]
    c       = inputs["c"]
    rho     = inputs["rho"]
    p0      = inputs["p0_smooth"]
    nt      = inputs["nt"]
    dt      = inputs["dt"]

    medium       = kWaveMedium(sound_speed=c, density=rho)
    source       = kSource()
    source.p0    = p0

    # Cartesian 2-point sensor: shape (1, 2) → columns are sensor x-coordinates [m].
    # Row 0 = x=-10mm, row 1 = x=+10mm after transposition by kspaceFirstOrder.
    sensor_mask_1d = np.array([[SENSOR_X_MM[0] * 1e-3, SENSOR_X_MM[1] * 1e-3]])
    sensor          = kSensor(mask=sensor_mask_1d)

    print(f"  [k-wave] Running 1-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        smooth_p0=True,   # already applied externally, but keep consistent
        pml_inside=True,
        backend="python",
        device="cpu",
        quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result["p"], dtype=np.float64)
    # Ensure shape (N_SENSORS, Nt)
    if pressure.ndim == 1:
        pressure = pressure.reshape(1, -1)
    if pressure.shape[0] != N_SENSORS:
        if pressure.shape[1] == N_SENSORS:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected k-wave sensor shape {pressure.shape}; "
                f"expected ({N_SENSORS}, {nt})"
            )

    _save_cache(_KWAVE_CACHE, pressure, nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    """Run pykwavers quasi-1D (Nx×1×1) with a 2-point grid sensor mask."""
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    c   = inputs["c"]
    rho = inputs["rho"]
    p0  = inputs["p0_smooth"]
    nt  = inputs["nt"]
    dt  = inputs["dt"]

    # Quasi-1D: Ny=Nz=1 so the solver reduces to 1-D propagation.
    grid = pkw.Grid(NX, 1, 1, DX, DX, DX)

    # Expand 1-D medium arrays to (Nx, 1, 1).
    c_3d   = c.reshape(NX, 1, 1).astype(np.float64)
    rho_3d = rho.reshape(NX, 1, 1).astype(np.float64)
    medium = pkw.Medium(sound_speed=c_3d, density=rho_3d)

    # Expand p0 to (Nx, 1, 1).
    p0_3d  = p0.reshape(NX, 1, 1)
    source = pkw.Source.from_initial_pressure(p0_3d)

    # Grid sensor mask (Nx, 1, 1): True at SENSOR_INDICES.
    # Fortran-order scan (x-fastest): index 56 → row 0, index 456 → row 1.
    # This matches k-wave's Cartesian ordering: −10 mm first, +10 mm second.
    sensor_mask = np.zeros((NX, 1, 1), dtype=bool)
    for idx in SENSOR_INDICES:
        sensor_mask[idx, 0, 0] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running quasi-1D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64)
    if pressure.ndim == 1:
        pressure = pressure.reshape(1, -1)
    if pressure.shape[0] != N_SENSORS:
        if pressure.shape[1] == N_SENSORS:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected pykwavers sensor shape {pressure.shape}; "
                f"expected ({N_SENSORS}, {nt})"
            )

    _save_cache(_PKWAV_CACHE, pressure, nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict, metrics: dict, *, status: str) -> None:
    kw_p  = kw["pressure"]    # (2, Nt)
    py_p  = pkw_res["pressure"]
    nt    = kw_p.shape[1]
    t_us  = np.arange(nt) * kw["dt"] * 1e6   # time axis in μs

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    sensor_labels = [f"x = {x:+.0f} mm" for x in SENSOR_X_MM]

    for row, (label, kw_trace, py_trace) in enumerate(
        zip(sensor_labels, kw_p, py_p)
    ):
        ax_t = axes[row, 0]   # time traces
        ax_e = axes[row, 1]   # error trace

        ax_t.plot(t_us, kw_trace, "b-", lw=0.8, label="k-wave-python")
        ax_t.plot(t_us, py_trace, "r--", lw=0.8, label="pykwavers")
        ax_t.set_title(f"Sensor {row + 1} ({label})")
        ax_t.set_xlabel("Time [μs]")
        ax_t.set_ylabel("Pressure [Pa]")
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.3)

        diff = py_trace - kw_trace
        ax_e.plot(t_us, diff, "g-", lw=0.6)
        ax_e.set_title(f"Difference (pykwavers − k-wave), sensor {row + 1}")
        ax_e.set_xlabel("Time [μs]")
        ax_e.set_ylabel("ΔPressure [Pa]")
        ax_e.axhline(0, color="k", lw=0.5)
        ax_e.grid(True, alpha=0.3)

    fig.suptitle(
        f"ivp_1D_simulation: k-wave-python vs pykwavers  [{status}]\n"
        f"Nx={NX}  dx={DX*1e3:.3f} mm  "
        f"c=[{C_SLOW},{C_FAST}] m/s  rho=[{RHO_LOW},{RHO_HIGH}] kg/m³  "
        f"pearson_r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for ivp_1D_simulation."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 70)
    print("ivp_1D_simulation: k-wave-python vs pykwavers")
    print(f"  Grid   : Nx={NX}  dx={DX*1e3:.4f} mm")
    print(f"  Medium : c=[{C_SLOW},{C_FAST}] m/s  rho=[{RHO_LOW},{RHO_HIGH}] kg/m³  "
          f"(piecewise heterogeneous, 1-D)")
    print(f"  Source : sinusoidal IVP pulse  x_pos={X_POS}  width={WIDTH}")
    print(f"  Sensor : 2-pt Cartesian at x={SENSOR_X_MM[0]:+.0f} mm and "
          f"x={SENSOR_X_MM[1]:+.0f} mm → grid indices {SENSOR_INDICES}")
    print(f"  t_end  : {T_END*1e6:.1f} μs  (2.5 × Nx×dx / c_fast)")
    print("=" * 70)

    print("\n[0/2] Building shared inputs...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    print(f"  Nt={nt}  dt={dt:.3e} s")

    # --- k-wave-python ---
    print("\n[1/2] k-wave-python (1-D PSTD, Cartesian 2-point sensor)...")
    kw = run_kwave(inputs, no_cache=no_cache)
    kw_p = kw["pressure"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(kw_p**2))):.4e} Pa")

    # --- pykwavers ---
    print("\n[2/2] pykwavers (quasi-1D PSTD, 2-point grid mask sensor)...")
    pkw_res = run_pykwavers(inputs, no_cache=no_cache)
    py_p = pkw_res["pressure"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(py_p**2))):.4e} Pa")

    # --- Per-sensor trace metrics ---
    print("\n--- Per-sensor trace metrics ---")
    sensor_metrics: list[dict] = []
    for row, label in enumerate([f"x={x:+.0f} mm" for x in SENSOR_X_MM]):
        m = compute_trace_metrics(kw_p[row], py_p[row])
        sensor_metrics.append(m)
        print(f"  Sensor {row + 1} ({label}):  "
              f"r={m['pearson_r']:.4f}  rms_ratio={m['rms_ratio']:.4f}  "
              f"rmse={m['rmse']:.3e} Pa")

    # --- Global matrix metrics (full 2×Nt flattened) ---
    metrics = compute_image_metrics(kw_p, py_p)

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r_min"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics["psnr_db"]   >= thr["psnr_db_min"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"\n--- Global parity metrics ---")
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r_min']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics['psnr_db']:.2f} dB  "
          f"(target >= {thr['psnr_db_min']} dB)  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  rmse      : {metrics['rmse']:.4e} Pa")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  "
          f"pykwavers={pkw_res['runtime_s']:.1f}s")

    # --- Figure ---
    plot_comparison(kw, pkw_res, metrics, status=status)

    # --- Text report ---
    header_lines = [
        "ivp_1D_simulation parity metrics",
        f"parity_status: {status}",
        f"grid: Nx={NX}  dx={DX:.6e} m",
        f"medium: c_fast={C_FAST} c_slow={C_SLOW} m/s  "
        f"rho_low={RHO_LOW} rho_high={RHO_HIGH} kg/m3  (piecewise 1D)",
        f"source: sinusoidal IVP  x_pos={X_POS}  width={WIDTH}  height={HEIGHT}  smooth_p0=True",
        f"sensor: Cartesian 2-pt at x={SENSOR_X_MM} mm → grid indices {SENSOR_INDICES}",
        f"t_end: {T_END:.6e} s  nt={nt}  dt={dt:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ]
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r_min']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f} dB  (target >= {thr['psnr_db_min']} dB)",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw_p).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(py_p).max()):.6e}",
        f"peak_ratio        = {metrics['peak_ratio']:.6f}",
    ]
    for row, label in enumerate([f"x={x:+.0f}mm" for x in SENSOR_X_MM]):
        m = sensor_metrics[row]
        report_lines += [
            f"sensor_{row+1}_{label}_pearson_r = {m['pearson_r']:.6f}",
            f"sensor_{row+1}_{label}_rms_ratio = {m['rms_ratio']:.6f}",
            f"sensor_{row+1}_{label}_rmse_Pa   = {m['rmse']:.6e}",
        ]
    save_text_report(METRICS_PATH, "\n".join(header_lines), report_lines)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
