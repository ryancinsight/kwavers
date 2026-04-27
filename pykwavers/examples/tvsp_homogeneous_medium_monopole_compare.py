#!/usr/bin/env python3
"""
tvsp_homogeneous_medium_monopole_compare.py
===========================================
Parity comparison for the upstream ``tvsp_homogeneous_medium_monopole`` example
(time-varying sinusoidal pressure source in a 2-D homogeneous absorbing medium).

Physical setup (matches k-wave-python ``examples/tvsp_homogeneous_medium_monopole.py``):
  Grid:    128×128  dx = dy = 50e-3/128 m ≈ 0.391 mm
  Medium:  homogeneous  c = 1500 m/s  rho = 1000 kg/m³
           absorption: alpha_coeff = 0.75 dB/(MHz^y cm)  alpha_power = 1.5
  Source:  single point at (95, 63) [0-based], filtered 0.25 MHz sinusoid
           (signal bandlimited by k-wave-python ``filter_time_series``)
  Sensor:  single point at (31, 63) [0-based], records pressure waveform
  PML:     20 grid points, inside domain

Comparison strategy
-------------------
Both engines are driven with the **same** physical parameters and the **same**
pre-filtered source signal (obtained from k-wave-python's ``setup()``).  This
isolates differences in the acoustic propagation solver from differences in
signal generation or filtering.

Outputs
-------
* ``output/tvsp_homogeneous_medium_monopole_compare.png``
* ``output/tvsp_homogeneous_medium_monopole_metrics.txt``

Usage
-----
  python examples/tvsp_homogeneous_medium_monopole_compare.py
  python examples/tvsp_homogeneous_medium_monopole_compare.py --no-cache
  python examples/tvsp_homogeneous_medium_monopole_compare.py --allow-failure
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
# Physical constants  (must match tvsp_homogeneous_medium_monopole.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 50e-3 / NX            # ≈ 3.906e-4 m

C0          = 1500.0             # sound speed [m/s]
RHO0        = 1000.0             # density [kg/m³]
ALPHA_COEFF = 0.75               # absorption coefficient [dB/(MHz^y cm)]
ALPHA_POWER = 1.5                # absorption power law exponent

SOURCE_IX   = 95                 # source grid index (0-based)
SOURCE_IY   = 63
SOURCE_FREQ = 0.25e6             # source frequency [Hz]
SOURCE_MAG  = 2.0                # source amplitude [Pa]

SENSOR_IX   = 31                 # sensor grid index (0-based)
SENSOR_IY   = 63

PML_SIZE    = 20                 # PML thickness [grid points]

# ---------------------------------------------------------------------------
# Parity thresholds
# A time-varying CW source with absorption generates a damped sinusoid at the
# sensor. Both solvers implement the same k-space PSTD scheme so high
# correlation is expected; the amplitude may differ slightly due to differences
# in PML formulation between k-wave-python (Chi/Psi CPML) and kwavers
# (fractional-Laplacian absorption), so the RMS ratio window is wider.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.97,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.25,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "tvsp_homogeneous_medium_monopole_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_homogeneous_medium_monopole_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_monopole_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_monopole_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Helper: cache I/O
# ---------------------------------------------------------------------------
def _load_kwave_cache() -> dict | None:
    if REFRESH_CACHE or not _KWAVE_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_KWAVE_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "trace":           np.asarray(d["trace"], dtype=np.float64).ravel(),
            "filtered_signal": np.asarray(d["filtered_signal"], dtype=np.float64).ravel(),
            "nt":              int(d["nt"]),
            "dt":              float(d["dt"]),
            "runtime_s":       float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_kwave_cache(
    trace: np.ndarray,
    filtered_signal: np.ndarray,
    nt: int,
    dt: float,
    runtime_s: float,
) -> None:
    _KWAVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        os.fspath(_KWAVE_CACHE),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        trace=np.asarray(trace, dtype=np.float64),
        filtered_signal=np.asarray(filtered_signal, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


def _load_pkwav_cache() -> dict | None:
    if REFRESH_CACHE or not _PKWAV_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_PKWAV_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "trace":     np.asarray(d["trace"], dtype=np.float64).ravel(),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_pkwav_cache(trace: np.ndarray, runtime_s: float) -> None:
    _PKWAV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        os.fspath(_PKWAV_CACHE),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        trace=np.asarray(trace, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Step 1 — k-wave-python reference run
# ---------------------------------------------------------------------------
def run_kwave(*, no_cache: bool = False) -> dict:
    """Run the upstream example and return the sensor trace plus the filtered source signal.

    Returns
    -------
    dict with keys:
        ``trace``           : 1-D float64 array of shape (Nt,) — sensor pressure [Pa]
        ``filtered_signal`` : 1-D float64 array of shape (Nt,) — filtered source signal [Pa]
        ``nt``              : int — number of time steps
        ``dt``              : float — time step size [s]
        ``runtime_s``       : float — wall-clock time [s]
    """
    if not no_cache:
        cached = _load_kwave_cache()
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    # Import the upstream example — uses the new kspaceFirstOrder API
    import sys as _sys
    from example_parity_utils import KWAVE_PYTHON_ROOT
    if str(KWAVE_PYTHON_ROOT / "examples") not in _sys.path:
        _sys.path.insert(0, str(KWAVE_PYTHON_ROOT / "examples"))

    from tvsp_homogeneous_medium_monopole import setup as kwave_setup, run as kwave_run

    print("  [k-wave] Building grid, medium, and filtered source signal...")
    kgrid, medium, source = kwave_setup()
    nt  = int(kgrid.Nt)
    dt  = float(kgrid.dt)
    filtered_signal = np.asarray(source.p, dtype=np.float64).ravel()

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kwave_run(backend="python", device="cpu", quiet=True)
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    # result["p"]: shape (1, Nt) — single sensor point
    trace = np.asarray(result["p"], dtype=np.float64).ravel()

    _save_kwave_cache(trace, filtered_signal, nt, dt, runtime_s)
    return {
        "trace":           trace,
        "filtered_signal": filtered_signal,
        "nt":              nt,
        "dt":              dt,
        "runtime_s":       runtime_s,
    }


# ---------------------------------------------------------------------------
# Step 2 — pykwavers CPU PSTD
# ---------------------------------------------------------------------------
def run_pykwavers(
    *,
    filtered_signal: np.ndarray,
    nt: int,
    dt: float,
    no_cache: bool = False,
) -> dict:
    """Run pykwavers with a binary mask source driven by the pre-filtered signal.

    Parameters
    ----------
    filtered_signal : 1-D float64 (Nt,) — bandlimited source signal [Pa]
    nt              : number of time steps
    dt              : time step [s]
    no_cache        : skip cache lookup

    Returns
    -------
    dict with keys ``trace`` (1-D float64) and ``runtime_s`` (float).
    """
    if not no_cache:
        cached = _load_pkwav_cache()
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    # --- Grid (quasi-2D: nz = 1) ---
    grid = pkw.Grid(NX, NY, 1, DX, DY, DX)

    # --- Medium: homogeneous with power-law absorption ---
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        absorption=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    # --- Source: single point at (SOURCE_IX, SOURCE_IY, 0), driven by filtered signal ---
    src_mask = np.zeros((NX, NY, 1), dtype=np.float64)
    src_mask[SOURCE_IX, SOURCE_IY, 0] = 1.0
    sig_1d = np.asarray(filtered_signal, dtype=np.float64).ravel()
    source = pkw.Source.from_mask(src_mask, sig_1d, SOURCE_FREQ)

    # --- Sensor: single point at (SENSOR_IX, SENSOR_IY, 0) ---
    sens_mask = np.zeros((NX, NY, 1), dtype=np.float64)
    sens_mask[SENSOR_IX, SENSOR_IY, 0] = 1.0
    sensor = pkw.Sensor.from_mask(sens_mask.astype(bool))

    # --- Simulation ---
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running CPU PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    trace = np.asarray(result.sensor_data, dtype=np.float64).ravel()
    _save_pkwav_cache(trace, runtime_s)
    return {"trace": trace, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    kw: dict,
    pkw_res: dict,
    metrics: dict[str, float],
    *,
    status: str,
) -> None:
    """Two-panel overlay: full trace + wavefront zoom."""
    t_us = np.arange(kw["nt"]) * kw["dt"] * 1e6  # µs
    kw_tr  = kw["trace"]
    pkw_tr = pkw_res["trace"]
    n = min(len(kw_tr), len(pkw_tr), len(t_us))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Full trace
    ax1.plot(t_us[:n], kw_tr[:n],  "k-",  linewidth=1.4, alpha=0.85, label="k-wave-python")
    ax1.plot(t_us[:n], pkw_tr[:n], "r--", linewidth=1.2, alpha=0.85, label="pykwavers")
    ax1.set_xlabel("Time [µs]")
    ax1.set_ylabel("Pressure [Pa]")
    ax1.set_title("Monopole sensor trace — full")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Zoom: last quarter of the simulation (steady-state propagating wave)
    i0 = max(0, n * 3 // 4)
    ax2.plot(t_us[i0:n], kw_tr[i0:n],  "k-",  linewidth=1.4, alpha=0.85, label="k-wave-python")
    ax2.plot(t_us[i0:n], pkw_tr[i0:n], "r--", linewidth=1.2, alpha=0.85, label="pykwavers")
    ax2.set_xlabel("Time [µs]")
    ax2.set_ylabel("Pressure [Pa]")
    ax2.set_title("Monopole sensor trace — late-time zoom")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"tvsp_homogeneous_medium_monopole: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}  dx={DX*1e3:.3f} mm  "
        f"source=[{SOURCE_IX},{SOURCE_IY}]  sensor=[{SENSOR_IX},{SENSOR_IY}]  "
        f"f₀={SOURCE_FREQ*1e-3:.0f} kHz  "
        f"r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}",
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
        description="Compare pykwavers with k-wave-python for tvsp_homogeneous_medium_monopole."
    )
    parser.add_argument("--no-cache",      action="store_true",
                        help="Force a fresh run (ignore cached NPZ files).")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 70)
    print("tvsp_homogeneous_medium_monopole: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}   dx={DX*1e3:.4f} mm")
    print(f"  Medium : c={C0} m/s  rho={RHO0} kg/m³  "
          f"alpha={ALPHA_COEFF} dB/(MHz^y cm)  y={ALPHA_POWER}")
    print(f"  Source : [{SOURCE_IX},{SOURCE_IY}]  f₀={SOURCE_FREQ*1e-3:.0f} kHz  "
          f"mag={SOURCE_MAG} Pa  filtered")
    print(f"  Sensor : [{SENSOR_IX},{SENSOR_IY}]  single point")
    print(f"  PML    : {PML_SIZE} pts inside")
    print("=" * 70)

    # --- k-wave-python ---
    print("\n[1/2] k-wave-python (new kspaceFirstOrder API, 2-D PSTD)...")
    kw = run_kwave(no_cache=no_cache)
    kw_tr = kw["trace"]
    print(f"  Nt={kw['nt']}  dt={kw['dt']:.3e} s  "
          f"peak={float(np.abs(kw_tr).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(kw_tr**2))):.4e} Pa")

    # --- pykwavers ---
    print("\n[2/2] pykwavers (CPU PSTD, Source.from_mask + pre-filtered signal)...")
    pkw_res = run_pykwavers(
        filtered_signal=kw["filtered_signal"],
        nt=kw["nt"],
        dt=kw["dt"],
        no_cache=no_cache,
    )
    pkw_tr = pkw_res["trace"]
    print(f"  peak={float(np.abs(pkw_tr).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(pkw_tr**2))):.4e} Pa")

    # --- Parity ---
    print("\n--- Parity evaluation ---")
    n = min(len(kw_tr), len(pkw_tr))
    metrics = compute_trace_metrics(kw_tr[:n], pkw_tr[:n])

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  rmse      : {metrics['rmse']:.4e} Pa")
    print(f"  peak_ratio: {metrics['peak_ratio']:.6f}")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    # --- Figure ---
    plot_comparison(kw, pkw_res, metrics, status=status)

    # --- Text report ---
    header_lines = [
        "tvsp_homogeneous_medium_monopole parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  rho={RHO0} kg/m3  "
        f"alpha_coeff={ALPHA_COEFF}  alpha_power={ALPHA_POWER}",
        f"source: [{SOURCE_IX},{SOURCE_IY}]  freq={SOURCE_FREQ:.3e} Hz  "
        f"mag={SOURCE_MAG} Pa  pre-filtered",
        f"sensor: [{SENSOR_IX},{SENSOR_IY}]  single point",
        f"pml_size: {PML_SIZE}  pml_inside: True",
        f"nt={kw['nt']}  dt={kw['dt']:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ]
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw_tr).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(pkw_tr).max()):.6e}",
        f"peak_ratio        = {metrics['peak_ratio']:.6f}",
    ]
    save_text_report(METRICS_PATH, "\n".join(header_lines), report_lines)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
