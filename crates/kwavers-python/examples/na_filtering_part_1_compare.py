#!/usr/bin/env python3
"""
na_filtering_part_1_compare.py
================================
Parity comparison for the upstream ``na_filtering_part_1`` example
(unfiltered delta-function source in a 1-D homogeneous medium).

Physical setup (matches k-wave-python ``examples/na_filtering_part_1.py``):
  Grid:    Nx=256  dx = 10e-3/256 ≈ 39.06 μm
  Medium:  homogeneous  c = 1500 m/s  (lossless)
  Source:  single grid point at index 50 (0-based)
           delta-function pulse: p[99] = 2 Pa  (time step 100, 1-based)
  Sensor:  full-grid binary mask (all 256 points), records pressure time-series
  Time:    dt = 7 ns  Nt = 1024  (fixed, not CFL-derived)
  PML:     inside domain

The original MATLAB example uses PMLInside=false to model a transparent
boundary with no absorption.  k-wave-python ports this with pml_inside=True
(the solver enforces the PML internally).  Both engines are driven with
identical parameters.

Parity note
-----------
The unfiltered delta-function excites all spatial frequencies, including those
above the grid's Nyquist limit.  Both solvers implement the same k-space
pseudospectral scheme so their broadband aliased outputs should be highly
correlated even in the presence of aliasing.  Comparison uses the full
(256 × 1024) sensor matrix.

1-D sensor ordering
-------------------
For a full-grid (256, 1, 1) mask, C-order and Fortran-order scans are
identical (the x-dimension is the only dimension).  No permutation is needed.

Grid cutoff frequency
---------------------
f_max = c / (2 × dx) = 1500 / (2 × 10e-3/256) ≈ 19.2 MHz.
This is used as the `frequency` metadata parameter for ``Source.from_mask``.

Outputs
-------
* ``output/na_filtering_part_1_compare.png``
* ``output/na_filtering_part_1_metrics.txt``

Usage
-----
  python examples/na_filtering_part_1_compare.py
  python examples/na_filtering_part_1_compare.py --no-cache
  python examples/na_filtering_part_1_compare.py --allow-failure
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
# Physical constants  (must match na_filtering_part_1.py exactly)
# ---------------------------------------------------------------------------
NX = 256
DX = 10e-3 / NX               # ≈ 39.0625e-6 m

C0         = 1500.0            # sound speed [m/s]
DT         = 7e-9              # time step [s]
NT         = 1024              # number of time steps

SOURCE_IDX      = 50           # source grid index (0-based)
TEMPORAL_OFFSET = 100          # delta spike at time step (1-based in MATLAB)
SOURCE_MAG      = 2.0          # [Pa]

SENSOR_ROW = 205               # sensor row for trace comparison (0-based)

# Grid frequency cutoff: f_max = c / (2 * dx)  ≈ 19.2 MHz
F_MAX = C0 / (2.0 * DX)

# ---------------------------------------------------------------------------
# Parity thresholds
# Broadband delta-function sources alias on both solvers.  The outputs should
# still be highly correlated since both implement the same PSTD scheme.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.95,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.25,
    "psnr_db":       15.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "na_filtering_part_1_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_filtering_part_1_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "na_filtering_part1_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "na_filtering_part1_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 2  # bumped: explicit pml_size=20 to match k-wave default


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
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_cache(path: os.PathLike, pressure: np.ndarray, runtime_s: float) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        pressure=np.asarray(pressure, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Source signal
# ---------------------------------------------------------------------------
def _build_source_signal() -> np.ndarray:
    """Delta-function pulse: p[TEMPORAL_OFFSET - 1] = SOURCE_MAG, rest 0."""
    sig = np.zeros(NT, dtype=np.float64)
    sig[TEMPORAL_OFFSET - 1] = SOURCE_MAG
    return sig


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(*, no_cache: bool = False) -> dict:
    """Run k-wave-python 1-D PSTD with unfiltered delta-function source."""
    if not no_cache:
        cached = _load_cache(_KWAVE_CACHE)
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid = kWaveGrid(Vector([NX]), Vector([DX]))
    kgrid.setTime(NT, DT)

    medium = kWaveMedium(sound_speed=C0)

    source        = kSource()
    p_mask        = np.zeros(NX, dtype=float)
    p_mask[SOURCE_IDX] = 1.0
    source.p_mask = p_mask
    source.p      = _build_source_signal()

    sensor = kSensor(mask=np.ones(NX, dtype=float), record=["p"])

    print(f"  [k-wave] Running 1-D PSTD  (Nt={NT}, dt={DT:.1e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        pml_inside=True, backend="python", device="cpu", quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result["p"], dtype=np.float64)   # (NX, NT)
    if pressure.shape[0] != NX:
        if pressure.shape[1] == NX:
            pressure = pressure.T
        else:
            raise AssertionError(f"Unexpected shape {pressure.shape}; expected ({NX}, {NT})")

    _save_cache(_KWAVE_CACHE, pressure, runtime_s)
    return {"pressure": pressure, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(*, no_cache: bool = False) -> dict:
    """Run pykwavers quasi-1D PSTD with the same unfiltered delta source."""
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    grid   = pkw.Grid(NX, 1, 1, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=1000.0)

    src_mask = np.zeros((NX, 1, 1), dtype=np.float64)
    src_mask[SOURCE_IDX, 0, 0] = 1.0
    source = pkw.Source.from_mask(src_mask, _build_source_signal(), F_MAX)

    # Full-grid sensor: all NX points, 1-D ordering is unambiguous.
    sens_mask = np.ones((NX, 1, 1), dtype=bool)
    sensor    = pkw.Sensor.from_mask(sens_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)
    sim.set_pml_size(20)  # match k-wave default (20 grid points)

    print(f"  [pykwavers] Running quasi-1D PSTD  (Nt={NT}, dt={DT:.1e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64)   # (NX, NT)
    if pressure.shape[0] != NX:
        if pressure.shape[1] == NX:
            pressure = pressure.T
        else:
            raise AssertionError(f"Unexpected shape {pressure.shape}; expected ({NX}, {NT})")

    _save_cache(_PKWAV_CACHE, pressure, runtime_s)
    return {"pressure": pressure, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict, metrics: dict, *, status: str) -> None:
    kw_p  = kw["pressure"]
    py_p  = pkw_res["pressure"]
    t_us  = np.arange(NT) * DT * 1e6
    sig   = _build_source_signal()

    # --- Spectrum helper ---
    f_hz  = np.fft.rfftfreq(NT, d=DT)
    f_mhz = f_hz * 1e-6

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Recorded traces at sensor row 205
    ax = axes[0, 0]
    ax.plot(t_us, kw_p[SENSOR_ROW], "b-",  lw=0.8, label="k-wave-python")
    ax.plot(t_us, py_p[SENSOR_ROW], "r--", lw=0.8, label="pykwavers")
    ax.set_title(f"Recorded trace at sensor row {SENSOR_ROW}")
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("Pressure [Pa]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Amplitude spectra
    ax = axes[0, 1]
    kw_as  = np.abs(np.fft.rfft(kw_p[SENSOR_ROW]))  / NT
    py_as  = np.abs(np.fft.rfft(py_p[SENSOR_ROW]))  / NT
    src_as = np.abs(np.fft.rfft(sig)) / NT
    ax.plot(f_mhz, src_as, "k-",  lw=0.8, label="Input spectrum")
    ax.plot(f_mhz, kw_as,  "b-",  lw=0.8, label="k-wave recorded")
    ax.plot(f_mhz, py_as,  "r--", lw=0.8, label="pykwavers recorded")
    ax.axvline(F_MAX * 1e-6, color="k", ls="--", lw=0.8, label=f"f_max={F_MAX*1e-6:.1f} MHz")
    ax.set_title("Amplitude spectra (sensor row 205)")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Full sensor matrix — k-wave
    ax = axes[1, 0]
    im = ax.imshow(kw_p, aspect="auto", origin="lower", cmap="seismic",
                   vmin=-float(np.abs(kw_p).max()), vmax=float(np.abs(kw_p).max()))
    ax.set_title("k-wave-python  full sensor matrix")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sensor index")
    fig.colorbar(im, ax=ax)

    # Full sensor matrix — pykwavers
    ax = axes[1, 1]
    im2 = ax.imshow(py_p, aspect="auto", origin="lower", cmap="seismic",
                    vmin=-float(np.abs(kw_p).max()), vmax=float(np.abs(kw_p).max()))
    ax.set_title("pykwavers  full sensor matrix")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sensor index")
    fig.colorbar(im2, ax=ax)

    fig.suptitle(
        f"na_filtering_part_1 (unfiltered delta): k-wave vs pykwavers  [{status}]\n"
        f"Nx={NX}  dx={DX*1e6:.1f} μm  c={C0} m/s  dt={DT*1e9:.0f} ns  Nt={NT}  "
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
        description="Compare pykwavers with k-wave-python for na_filtering_part_1."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("na_filtering_part_1: k-wave-python vs pykwavers")
    print(f"  Grid   : Nx={NX}  dx={DX*1e6:.2f} μm")
    print(f"  Medium : c={C0} m/s  (lossless)")
    print(f"  Source : delta-function at idx={SOURCE_IDX}  t_step={TEMPORAL_OFFSET}  "
          f"mag={SOURCE_MAG} Pa")
    print(f"  Sensor : full-grid ({NX} pts)")
    print(f"  Time   : dt={DT*1e9:.0f} ns  Nt={NT}")
    print("=" * 70)

    print("\n[1/2] k-wave-python (1-D PSTD, unfiltered delta source)...")
    kw = run_kwave(no_cache=args.no_cache)
    kw_p = kw["pressure"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa")

    print("\n[2/2] pykwavers (quasi-1D PSTD, same unfiltered source)...")
    pkw_res = run_pykwavers(no_cache=args.no_cache)
    py_p = pkw_res["pressure"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa")

    # --- Parity on full matrix ---
    metrics = compute_image_metrics(kw_p, py_p)

    # --- Per-trace at sensor row 205 ---
    trace_m = compute_trace_metrics(kw_p[SENSOR_ROW], py_p[SENSOR_ROW])

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics["psnr_db"]   >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"\n--- Parity (full sensor matrix) ---")
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics['psnr_db']:.2f} dB  "
          f"(target >= {thr['psnr_db']} dB)  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  Sensor row {SENSOR_ROW}: r={trace_m['pearson_r']:.4f}  "
          f"rms_ratio={trace_m['rms_ratio']:.4f}")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  "
          f"pykwavers={pkw_res['runtime_s']:.1f}s")

    plot_comparison(kw, pkw_res, metrics, status=status)

    header = "\n".join([
        "na_filtering_part_1 parity metrics",
        f"parity_status: {status}",
        f"grid: Nx={NX}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  lossless",
        f"source: delta-function  idx={SOURCE_IDX}  t_step={TEMPORAL_OFFSET}  "
        f"mag={SOURCE_MAG} Pa",
        f"sensor: full-grid ({NX} pts)",
        f"dt={DT:.3e} s  nt={NT}",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ])
    report = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f} dB  (target >= {thr['psnr_db']} dB)",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"sensor_row_{SENSOR_ROW}_pearson_r = {trace_m['pearson_r']:.6f}",
        f"sensor_row_{SENSOR_ROW}_rms_ratio = {trace_m['rms_ratio']:.6f}",
    ]
    save_text_report(METRICS_PATH, header, report)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
