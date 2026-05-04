#!/usr/bin/env python3
"""
tvsp_homogeneous_medium_dipole_compare.py
==========================================
Parity comparison for the upstream ``tvsp_homogeneous_medium_dipole`` example
(dipole velocity source in a 2-D homogeneous absorbing medium).

Physical setup (matches k-wave-python ``examples/tvsp_homogeneous_medium_dipole.py``):
  Grid:    128×128   dx = dy = 50 mm / 128 ≈ 0.390625 mm
  Medium:  homogeneous, lossless
             c   = 1500 m/s
             rho = 1000 kg/m³
             alpha_coeff = 0.75 dB/(MHz^y cm)
             alpha_power = 1.5
  Source:  particle-velocity dipole at (95, 63) 0-based (MATLAB: (96, 64) 1-based)
           ux = -source_mag * sin(2π * 0.25 MHz * t), filtered to grid Nyquist
           source_mag = 2 / (c * rho)
  Sensor:  single point at (31, 63) 0-based (MATLAB: (32, 64) 1-based)
  PML:     default (inside)

Dipole source in pykwavers
--------------------------
``pkw.Source.from_velocity_mask(mask_3d, ux=ux_1d)`` injects particle velocity
in the x-direction at the active mask point.  The k-wave and pykwavers engines
share the same filtered ux signal (computed once from k-wave's filter_time_series).

Outputs
-------
* ``output/tvsp_homogeneous_medium_dipole_compare.png``
* ``output/tvsp_homogeneous_medium_dipole_metrics.txt``

Usage
-----
  python examples/tvsp_homogeneous_medium_dipole_compare.py
  python examples/tvsp_homogeneous_medium_dipole_compare.py --no-cache
  python examples/tvsp_homogeneous_medium_dipole_compare.py --allow-failure
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
    compute_trace_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical constants  (must match tvsp_homogeneous_medium_dipole.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 50e-3 / NX           # ≈ 0.390625 mm

C0          = 1500.0            # [m/s]
RHO0        = 1000.0            # [kg/m³]
ALPHA_COEFF = 0.75              # [dB/(MHz^y cm)]
ALPHA_POWER = 1.5

SOURCE_FREQ = 0.25e6            # [Hz]
# source_mag = 2 / (c * rho) matches the MATLAB example
SOURCE_MAG  = 2.0 / (C0 * RHO0)

# 0-based positions
SOURCE_X = 95
SOURCE_Y = 63
SENSOR_X = 31
SENSOR_Y = 63

# ---------------------------------------------------------------------------
# Parity thresholds
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.90,
    "rms_ratio_min": 0.75,
    "rms_ratio_max": 1.30,
    "psnr_db":       15.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "tvsp_homogeneous_medium_dipole_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_homogeneous_medium_dipole_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_dipole_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_dipole_pykwavers_cache.npz"

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
            "pressure":  np.asarray(d["pressure"],  dtype=np.float64),
            "nt":        int(d["nt"]),
            "dt":        float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_cache(path: os.PathLike, pressure: np.ndarray, nt: int,
                dt: float, runtime_s: float) -> None:
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
def build_shared_inputs() -> dict:
    """Compute grid, medium, and shared filtered ux signal."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.filters import filter_time_series

    kgrid  = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    medium = kWaveMedium(
        sound_speed=C0,
        density=RHO0,
        alpha_coeff=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )
    kgrid.makeTime(medium.sound_speed)

    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    t_array = np.asarray(kgrid.t_array).ravel()

    # Dipole: x-velocity source
    ux_raw = -SOURCE_MAG * np.sin(2 * np.pi * SOURCE_FREQ * t_array)
    ux_filt = np.asarray(
        filter_time_series(kgrid, medium, ux_raw.reshape(1, -1)),
        dtype=np.float64,
    ).ravel()   # shape (Nt,)

    # Single-point velocity mask
    u_mask_2d = np.zeros((NX, NY), dtype=np.float64)
    u_mask_2d[SOURCE_X, SOURCE_Y] = 1.0

    # Single-point sensor mask
    sensor_mask_2d = np.zeros((NX, NY), dtype=np.float64)
    sensor_mask_2d[SENSOR_X, SENSOR_Y] = 1.0

    return {
        "kgrid":          kgrid,
        "medium":         medium,
        "ux_filt":        ux_filt,
        "u_mask_2d":      u_mask_2d,
        "sensor_mask_2d": sensor_mask_2d,
        "nt":             nt,
        "dt":             dt,
    }


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_KWAVE_CACHE)
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid  = inputs["kgrid"]
    medium = inputs["medium"]
    nt     = inputs["nt"]
    dt     = inputs["dt"]

    source        = kSource()
    source.u_mask = inputs["u_mask_2d"]
    # k-wave expects (num_sources, Nt) for ux; single source → (1, Nt)
    source.ux     = inputs["ux_filt"].reshape(1, -1)

    sensor_mask_2d = inputs["sensor_mask_2d"]
    sensor = kSensor(mask=sensor_mask_2d, record=["p"])

    print(f"  [k-wave] Running 2-D PSTD dipole  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        pml_inside=True,
        backend="python",
        device="cpu",
        quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    # Single sensor point: shape is (1, Nt) or (Nt,)
    pressure = np.asarray(result["p"], dtype=np.float64).ravel()

    _save_cache(_KWAVE_CACHE, pressure.reshape(1, -1), nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    nt = inputs["nt"]
    dt = inputs["dt"]

    grid   = pkw.Grid(NX, NY, 1, DX, DY, DX)
    medium = pkw.Medium(
        sound_speed=C0,
        density=RHO0,
        alpha_coeff=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    # Expand velocity mask to 3D (NX, NY, 1)
    u_mask_3d = inputs["u_mask_2d"][:, :, None]
    ux_1d     = inputs["ux_filt"]   # shape (Nt,)

    source = pkw.Source.from_velocity_mask(u_mask_3d, ux=ux_1d)

    # Single-point sensor: (NX, NY, 1) bool mask
    sensor_mask_3d = inputs["sensor_mask_2d"][:, :, None].astype(bool)
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running quasi-2D PSTD dipole  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64).ravel()

    _save_cache(_PKWAV_CACHE, pressure.reshape(1, -1), nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict, metrics: dict, *,
                    status: str) -> None:
    kw_p  = kw["pressure"]
    py_p  = pkw_res["pressure"]
    nt    = kw["nt"]
    dt    = kw["dt"]
    t_us  = np.arange(nt) * dt * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_us, kw_p, "b-",  lw=0.8, label="k-wave-python")
    axes[0].plot(t_us, py_p, "r--", lw=0.8, label="pykwavers")
    axes[0].set_title(f"Sensor ({SENSOR_X},{SENSOR_Y}) pressure")
    axes[0].set_xlabel("Time [μs]")
    axes[0].set_ylabel("Pressure [Pa]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_us, py_p - kw_p, "k-", lw=0.8)
    axes[1].set_title("Residual: pykwavers − k-wave")
    axes[1].set_xlabel("Time [μs]")
    axes[1].set_ylabel("ΔPressure [Pa]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"tvsp_homogeneous_medium_dipole: k-wave vs pykwavers  [{status}]\n"
        f"{NX}×{NY}  dx={DX*1e3:.3f} mm  f={SOURCE_FREQ/1e6:.2f} MHz  "
        f"ux-dipole at ({SOURCE_X},{SOURCE_Y})\n"
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
        description="Compare pykwavers with k-wave-python for tvsp_homogeneous_medium_dipole."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("tvsp_homogeneous_medium_dipole: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}  dx={DX*1e3:.3f} mm")
    print(f"  Source : ux-dipole at ({SOURCE_X},{SOURCE_Y})  f={SOURCE_FREQ/1e6:.2f} MHz")
    print(f"  Sensor : single point ({SENSOR_X},{SENSOR_Y})")
    print("=" * 70)

    print("\n[0/2] Building shared inputs...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    print(f"  Nt={nt}  dt={dt:.3e} s")

    print("\n[1/2] k-wave-python (2-D PSTD, velocity dipole)...")
    kw = run_kwave(inputs, no_cache=args.no_cache)
    kw_p = kw["pressure"]
    print(f"  Nt={nt}  peak={float(np.abs(kw_p).max()):.4e} Pa")

    print("\n[2/2] pykwavers (quasi-2D PSTD, velocity dipole)...")
    pkw_res = run_pykwavers(inputs, no_cache=args.no_cache)
    py_p = pkw_res["pressure"]
    print(f"  Nt={nt}  peak={float(np.abs(py_p).max()):.4e} Pa")

    metrics = compute_trace_metrics(kw_p, py_p)

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"\n--- Parity ---")
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  {'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    plot_comparison(kw, pkw_res, metrics, status=status)

    header = "\n".join([
        "tvsp_homogeneous_medium_dipole parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  rho={RHO0} kg/m3  alpha={ALPHA_COEFF}  power={ALPHA_POWER}",
        f"source: ux-dipole at ({SOURCE_X},{SOURCE_Y})  f={SOURCE_FREQ:.3e} Hz",
        f"sensor: ({SENSOR_X},{SENSOR_Y})",
        f"nt={nt}  dt={dt:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ])
    report = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  (target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw_p).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(py_p).max()):.6e}",
    ]
    save_text_report(METRICS_PATH, header, report)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
