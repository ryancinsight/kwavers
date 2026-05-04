#!/usr/bin/env python3
"""
na_filtering_part_2_compare.py
================================
Parity comparison for the upstream ``na_filtering_part_2`` example
(spatially smoothed source mask reduces aliasing in a 1-D medium).

Physical setup (matches k-wave-python ``examples/na_filtering_part_2.py``):
  Grid:    Nx=256  dx = 10e-3/256 ≈ 39.06 μm
  Medium:  homogeneous  c = 1500 m/s  (lossless)
  Source:  spatially smoothed mask near index 50 (0-based);
           each active element is weighted by its smoothed magnitude and
           driven by the same delta-function pulse (p[99] = 2 Pa).
           Elements below threshold 0.05 are zeroed; mask is then binarized.
  Sensor:  full-grid binary mask (all 256 points)
  Time:    dt = 7 ns  Nt = 1024  (fixed)
  PML:     inside domain

Spatial smoothing
-----------------
k-wave's ``smooth(mask, restore_max=True)`` applies a Blackman-windowed
k-space filter and rescales to unit maximum.  The smoothed mask is
thresholded at 0.05, binarized, and the per-element weights are folded into
the source time signal matrix ``source.p = weights[:, None] * delta[None, :]``
with shape (num_active, Nt).

pykwavers ``Source.from_mask`` accepts a 2D signal matrix [num_sources × Nt]
when the mask has multiple active elements.

Outputs
-------
* ``output/na_filtering_part_2_compare.png``
* ``output/na_filtering_part_2_metrics.txt``

Usage
-----
  python examples/na_filtering_part_2_compare.py
  python examples/na_filtering_part_2_compare.py --no-cache
  python examples/na_filtering_part_2_compare.py --allow-failure
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
# Physical constants  (must match na_filtering_part_2.py exactly)
# ---------------------------------------------------------------------------
NX = 256
DX = 10e-3 / NX

C0         = 1500.0
DT         = 7e-9
NT         = 1024

SOURCE_IDX      = 50
TEMPORAL_OFFSET = 100
SOURCE_MAG      = 2.0
SMOOTH_THRESH   = 0.05          # threshold for removing small smoothed values

SENSOR_ROW = 205
F_MAX      = C0 / (2.0 * DX)   # ≈ 19.2 MHz — metadata for Source.from_mask

# ---------------------------------------------------------------------------
# Parity thresholds
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
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "na_filtering_part_2_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_filtering_part_2_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "na_filtering_part2_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "na_filtering_part2_pykwavers_cache.npz"

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
# Shared source preparation
# ---------------------------------------------------------------------------
def _build_smoothed_source() -> tuple[np.ndarray, np.ndarray]:
    """Apply Blackman-windowed spatial smoothing and return (binary_mask_1d, signal_2d).

    Returns
    -------
    binary_mask : (NX,) float64 — 0/1 binary source mask
    signal_2d  : (num_active, NT) float64 — per-element scaled delta pulses
    """
    from kwave.utils.filters import smooth

    p_mask = np.zeros((NX, 1), dtype=float)
    p_mask[SOURCE_IDX, 0] = 1.0
    p_mask = smooth(p_mask, restore_max=True)
    p_mask[p_mask < SMOOTH_THRESH] = 0.0
    p_mask = p_mask.ravel()

    nonzero_vals = p_mask[p_mask != 0.0]

    delta = np.zeros(NT, dtype=float)
    delta[TEMPORAL_OFFSET - 1] = SOURCE_MAG
    signal_2d = nonzero_vals[:, np.newaxis] * delta[np.newaxis, :]   # (num_active, NT)

    # Binarize mask
    binary_mask = np.where(p_mask != 0.0, 1.0, 0.0)
    return binary_mask, signal_2d


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(*, no_cache: bool = False) -> dict:
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

    binary_mask, signal_2d = _build_smoothed_source()

    source        = kSource()
    source.p_mask = binary_mask
    source.p      = signal_2d                      # (num_active, NT)

    sensor = kSensor(mask=np.ones(NX, dtype=float), record=["p"])

    print(f"  [k-wave] Running 1-D PSTD  (Nt={NT}, dt={DT:.1e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        pml_inside=True, backend="python", device="cpu", quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result["p"], dtype=np.float64)
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
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    binary_mask, signal_2d = _build_smoothed_source()

    grid   = pkw.Grid(NX, 1, 1, DX, DX, DX)
    medium = pkw.Medium(sound_speed=C0, density=1000.0)

    # Expand binary mask to (NX, 1, 1); signal_2d is (num_active, NT).
    src_mask_3d = binary_mask.reshape(NX, 1, 1).astype(np.float64)
    source = pkw.Source.from_mask(src_mask_3d, signal_2d, F_MAX)

    sens_mask = np.ones((NX, 1, 1), dtype=bool)
    sensor    = pkw.Sensor.from_mask(sens_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running quasi-1D PSTD  (Nt={NT}, dt={DT:.1e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64)
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(t_us, kw_p[SENSOR_ROW], "b-",  lw=0.8, label="k-wave-python")
    ax.plot(t_us, py_p[SENSOR_ROW], "r--", lw=0.8, label="pykwavers")
    ax.set_title(f"Sensor row {SENSOR_ROW} (spatially smoothed)")
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("Pressure [Pa]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    vmax = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))
    im1 = axes[1].imshow(kw_p, aspect="auto", origin="lower", cmap="seismic",
                         vmin=-vmax, vmax=vmax)
    axes[1].set_title("k-wave-python")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Sensor index")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(py_p, aspect="auto", origin="lower", cmap="seismic",
                         vmin=-vmax, vmax=vmax)
    axes[2].set_title("pykwavers")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Sensor index")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(
        f"na_filtering_part_2 (smoothed mask): k-wave vs pykwavers  [{status}]\n"
        f"Nx={NX}  c={C0} m/s  threshold={SMOOTH_THRESH}  "
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
        description="Compare pykwavers with k-wave-python for na_filtering_part_2."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("na_filtering_part_2: k-wave-python vs pykwavers")
    print(f"  Grid   : Nx={NX}  dx={DX*1e6:.2f} μm")
    print(f"  Source : smoothed mask (Blackman filter) near idx={SOURCE_IDX}  "
          f"threshold={SMOOTH_THRESH}")
    print("=" * 70)

    print("\n[1/2] k-wave-python (1-D PSTD, spatially smoothed source)...")
    kw = run_kwave(no_cache=args.no_cache)
    kw_p = kw["pressure"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa")

    print("\n[2/2] pykwavers (quasi-1D PSTD, smoothed source matrix)...")
    pkw_res = run_pykwavers(no_cache=args.no_cache)
    py_p = pkw_res["pressure"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa")

    metrics  = compute_image_metrics(kw_p, py_p)
    trace_m  = compute_trace_metrics(kw_p[SENSOR_ROW], py_p[SENSOR_ROW])

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics["psnr_db"]   >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"\n--- Parity ---")
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  {'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics['psnr_db']:.2f} dB  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  Sensor row {SENSOR_ROW}: r={trace_m['pearson_r']:.4f}  "
          f"rms_ratio={trace_m['rms_ratio']:.4f}")

    plot_comparison(kw, pkw_res, metrics, status=status)

    header = "\n".join([
        "na_filtering_part_2 parity metrics",
        f"parity_status: {status}",
        f"grid: Nx={NX}  dx={DX:.6e} m  c={C0} m/s",
        f"source: smoothed mask near idx={SOURCE_IDX}  threshold={SMOOTH_THRESH}",
        f"dt={DT:.3e} s  nt={NT}",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ])
    report = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}",
        f"psnr_db    = {metrics['psnr_db']:.2f} dB",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"sensor_row_{SENSOR_ROW}_pearson_r = {trace_m['pearson_r']:.6f}",
        f"sensor_row_{SENSOR_ROW}_rms_ratio = {trace_m['rms_ratio']:.6f}",
    ]
    save_text_report(METRICS_PATH, header, report)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
