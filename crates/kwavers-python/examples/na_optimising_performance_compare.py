#!/usr/bin/env python3
"""
na_optimising_performance_compare.py
======================================
Parity comparison for the upstream ``na_optimising_performance`` example
(image-derived initial pressure distribution in a 2-D homogeneous medium
with a Cartesian circular sensor; originally framed around DataCast
performance tuning which is irrelevant for the Python backend).

Physical setup (matches k-wave-python ``examples/na_optimising_performance.py``):
  Grid    : 256×256  dx = dy = 10 mm / 256 ≈ 39.1 µm
  Medium  : homogeneous  c = 1500 m/s, rho = 1000 kg/m³
  Source  : initial pressure from EXAMPLE_source_two.bmp
            p0_magnitude = 2 Pa  (scaled after resize to 256×256)
            smooth_p0 = True (k-wave default)
  Sensor  : Cartesian circle, r = 4.5 mm, 100 points → grid mask
  Records : p (time-series) and p_final (end-state 2-D field)
  PML     : default (auto) — k-wave-python uses pml_inside=True default

Source image: ``external/k-wave-python/tests/EXAMPLE_source_two.bmp``
Must be present; the upstream k-wave-python test suite ships it.

Outputs
-------
* ``output/na_optimising_performance_compare.png``
* ``output/na_optimising_performance_metrics.txt``

Usage
-----
  python examples/na_optimising_performance_compare.py
  python examples/na_optimising_performance_compare.py --no-cache
  python examples/na_optimising_performance_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical constants (matches na_optimising_performance.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 256
X = Y = 10e-3   # [m]
DX = X / NX     # ≈ 3.906e-5 m
DY = Y / NY

C0  = 1500.0    # [m/s]
RHO = 1000.0    # [kg/m³]

P0_MAGNITUDE = 2.0   # [Pa]

SENSOR_RADIUS     = 4.5e-3   # [m]
NUM_SENSOR_POINTS = 100

PML_SIZE = 20   # k-wave-python default for 256×256 grids

# Image source (shipped with k-wave-python tests)
SOURCE_IMAGE_PATH = (
    Path(__file__).parents[3]
    / "external"
    / "k-wave-python"
    / "tests"
    / "EXAMPLE_source_two.bmp"
)

# ---------------------------------------------------------------------------
# Parity thresholds
# Homogeneous medium + image source: both engines should converge closely.
# The image source distributes energy across the full grid, so some PSTD
# dispersion drift is expected in the trailing wavefronts.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.85,
    "rms_ratio_min": 0.70,
    "rms_ratio_max": 1.40,
    "psnr_db":       12.0,
}

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "na_optimising_performance_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_optimising_performance_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "na_optimising_performance_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "na_optimising_performance_pykwavers_cache.npz"

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
            "p_final":   np.asarray(d["p_final"],   dtype=np.float64),
            "nt":        int(d["nt"]),
            "dt":        float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
            "n_sensors": int(d["n_sensors"]),
        }
    except Exception:
        return None


def _save_cache(
    path: os.PathLike,
    pressure: np.ndarray,
    p_final: np.ndarray,
    nt: int,
    dt: float,
    runtime_s: float,
    n_sensors: int,
) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        pressure=np.asarray(pressure, dtype=np.float64),
        p_final=np.asarray(p_final,  dtype=np.float64),
        nt=np.array(nt,        dtype=np.int64),
        dt=np.array(dt,        dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
        n_sensors=np.array(n_sensors, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
def build_shared_inputs() -> dict:
    """Construct grid, homogeneous medium, image-derived p0, and grid sensor mask."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.io import load_image
    from kwave.utils.mapgen import make_cart_circle
    from kwave.utils.matrix import resize
    from kwave.utils.conversion import cart2grid as kwave_cart2grid
    from kwave.utils.filters import smooth as kwave_smooth

    if not SOURCE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Source image not found: {SOURCE_IMAGE_PATH}\n"
            "Expected in external/k-wave-python/tests/EXAMPLE_source_two.bmp"
        )

    # Load and resize image
    raw = load_image(str(SOURCE_IMAGE_PATH), is_gray=True)
    p0_raw = P0_MAGNITUDE * resize(raw, [NX, NY])
    p0 = np.asarray(p0_raw, dtype=np.float64)

    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    kgrid.makeTime(C0)

    # Shared smoothed p0 (k-wave default smooth_p0=True)
    p0_smooth = np.asarray(kwave_smooth(p0, restore_max=True), dtype=np.float64)

    # Cartesian circle → C-order grid mask
    sensor_circle = make_cart_circle(SENSOR_RADIUS, NUM_SENSOR_POINTS)
    sensor_mask_2d, _, _ = kwave_cart2grid(kgrid, sensor_circle, order="C")
    sensor_mask_2d = np.asarray(sensor_mask_2d, dtype=bool)
    n_sensors = int(sensor_mask_2d.sum())

    # C → Fortran sensor-row permutation
    active_c = np.argwhere(sensor_mask_2d)
    sensor_row_perm = np.lexsort((active_c[:, 0], active_c[:, 1]))

    return {
        "kgrid":           kgrid,
        "p0":              p0,
        "p0_smooth":       p0_smooth,
        "sensor_mask_2d":  sensor_mask_2d,
        "n_sensors":       n_sensors,
        "sensor_row_perm": sensor_row_perm,
        "nt":              int(kgrid.Nt),
        "dt":              float(kgrid.dt),
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
    from kwave.kmedium import kWaveMedium
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid     = inputs["kgrid"]
    nt        = inputs["nt"]
    dt        = inputs["dt"]
    n_sensors = inputs["n_sensors"]

    medium  = kWaveMedium(sound_speed=C0)
    source  = kSource()
    source.p0 = inputs["p0"]   # k-wave applies smooth_p0=True internally

    sensor = kSensor(mask=inputs["sensor_mask_2d"])
    sensor.record = ["p", "p_final"]

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        smooth_p0=True,
        pml_inside=True,
        pml_size=PML_SIZE,
        backend="python",
        device="cpu",
        quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result["p"], dtype=np.float64)
    p_final  = np.asarray(result["p_final"], dtype=np.float64).squeeze()

    if pressure.shape[0] != n_sensors:
        if pressure.shape[1] == n_sensors:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"k-wave pressure shape {pressure.shape}; expected ({n_sensors}, {nt})"
            )

    _save_cache(_KWAVE_CACHE, pressure, p_final, nt, dt, runtime_s, n_sensors)
    return {"pressure": pressure, "p_final": p_final,
            "nt": nt, "dt": dt, "runtime_s": runtime_s, "n_sensors": n_sensors}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    nt        = inputs["nt"]
    dt        = inputs["dt"]
    n_sensors = inputs["n_sensors"]
    p0_smooth = inputs["p0_smooth"]

    grid   = pkw.Grid(NX, NY, 1, DX, DY, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO)

    # Use the pre-smoothed p0 (same one k-wave sees after smooth_p0=True)
    p0_3d  = p0_smooth[:, :, None]
    source = pkw.Source.from_initial_pressure(p0_3d)

    sensor_mask_3d = inputs["sensor_mask_2d"][:, :, None]
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)
    sensor.set_record(["p", "p_final"])

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running CPU PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64)
    if pressure.shape[0] != n_sensors:
        if pressure.shape[1] == n_sensors:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"pykwavers pressure shape {pressure.shape}; expected ({n_sensors}, {nt})"
            )

    pf = result.p_final_field
    if pf is not None:
        p_final_3d = np.asarray(pf, dtype=np.float64)
        p_final = p_final_3d[:, :, 0] if p_final_3d.ndim == 3 else p_final_3d.squeeze()
    else:
        p_final = np.zeros((NX, NY), dtype=np.float64)

    _save_cache(_PKWAV_CACHE, pressure, p_final, nt, dt, runtime_s, n_sensors)
    return {"pressure": pressure, "p_final": p_final,
            "nt": nt, "dt": dt, "runtime_s": runtime_s, "n_sensors": n_sensors}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    kw: dict,
    pkw_res: dict,
    inputs: dict,
    metrics_p: dict,
    metrics_pf: dict,
    *,
    status: str,
) -> None:
    kw_p   = kw["pressure"]
    py_p   = pkw_res["pressure"]
    diff_p = py_p - kw_p
    vmax_p = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))
    dmax_p = float(max(np.abs(diff_p).max(), 1e-30))

    kw_pf   = kw["p_final"]
    py_pf   = pkw_res["p_final"]
    diff_pf = py_pf - kw_pf
    vmax_pf = float(max(np.abs(kw_pf).max(), np.abs(py_pf).max(), 1e-30))
    dmax_pf = float(max(np.abs(diff_pf).max(), 1e-30))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 0: sensor time-series comparison
    for ax, data, title in zip(
        axes[0],
        [kw_p, py_p, diff_p],
        ["k-wave-python (sensor)", "pykwavers (sensor)", "Diff sensor"],
    ):
        vmax = dmax_p if "Diff" in title else vmax_p
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="seismic",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Sensor index")
        fig.colorbar(im, ax=ax)

    # Row 1: p_final field comparison
    for ax, data, title in zip(
        axes[1],
        [kw_pf, py_pf, diff_pf],
        ["k-wave-python (p_final)", "pykwavers (p_final)", "Diff p_final"],
    ):
        vmax = dmax_pf if "Diff" in title else vmax_pf
        im = ax.imshow(data.T, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax)

    # Overlay source mask on p0 for reference
    sensor_mask = inputs["sensor_mask_2d"]
    sens_y, sens_x = np.where(sensor_mask)
    for ax in axes[1, :2]:
        ax.plot(sens_x, sens_y, "k.", markersize=1, alpha=0.5)

    fig.suptitle(
        f"na_optimising_performance: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}  dx={DX*1e6:.2f} µm  "
        f"sensor_r={metrics_p['pearson_r']:.4f}  "
        f"p_final_r={metrics_pf['pearson_r']:.4f}",
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
        description="Compare pykwavers with k-wave-python for na_optimising_performance."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    no_cache = args.no_cache

    if not SOURCE_IMAGE_PATH.exists():
        print(f"ERROR: Source image not found: {SOURCE_IMAGE_PATH}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("na_optimising_performance: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}   dx={DX*1e6:.2f} µm")
    print(f"  Medium : c={C0} m/s  rho={RHO} kg/m³ (homogeneous)")
    print(f"  Source : EXAMPLE_source_two.bmp → p0_magnitude={P0_MAGNITUDE} Pa")
    print(f"  Sensor : {NUM_SENSOR_POINTS}-pt Cartesian circle r={SENSOR_RADIUS*1e3:.1f} mm")
    print(f"  PML    : {PML_SIZE} pts inside")
    print("=" * 70)

    print("\n[0/2] Building shared inputs (grid, p0 from image, sensor mask)...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    n_sensors = inputs["n_sensors"]
    p0 = inputs["p0"]
    print(f"  Nt={nt}  dt={dt:.3e} s  n_sensors={n_sensors}")
    print(f"  p0: shape={p0.shape}  peak={float(np.abs(p0).max()):.4e} Pa")

    print("\n[1/2] k-wave-python (2-D PSTD)...")
    kw = run_kwave(inputs, no_cache=no_cache)
    print(f"  sensor: shape={kw['pressure'].shape}  "
          f"peak={float(np.abs(kw['pressure']).max()):.4e} Pa")
    print(f"  p_final: shape={kw['p_final'].shape}  "
          f"peak={float(np.abs(kw['p_final']).max()):.4e} Pa")

    print("\n[2/2] pykwavers (CPU PSTD)...")
    pkw_res = run_pykwavers(inputs, no_cache=no_cache)
    print(f"  sensor: shape={pkw_res['pressure'].shape}  "
          f"peak={float(np.abs(pkw_res['pressure']).max()):.4e} Pa")
    print(f"  p_final: shape={pkw_res['p_final'].shape}  "
          f"peak={float(np.abs(pkw_res['p_final']).max()):.4e} Pa")

    # Align sensor row ordering (C → Fortran)
    perm = inputs["sensor_row_perm"]
    kw_p_aligned = kw["pressure"][perm]

    print("\n--- Parity evaluation ---")
    metrics_p  = compute_image_metrics(kw_p_aligned, pkw_res["pressure"])
    metrics_pf = compute_image_metrics(kw["p_final"], pkw_res["p_final"])

    thr = PARITY_THRESHOLDS
    checks_p = {
        "pearson_r": metrics_p["pearson_r"] >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics_p["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics_p["psnr_db"]  >= thr["psnr_db"],
    }
    checks_pf = {
        "pearson_r": metrics_pf["pearson_r"] >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics_pf["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics_pf["psnr_db"]  >= thr["psnr_db"],
    }
    status = "PASS" if (all(checks_p.values()) and all(checks_pf.values())) else "FAIL"

    print(f"\n  [sensor time-series]")
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics_p['pearson_r']:.6f}  (>= {thr['pearson_r']})  "
          f"{'OK' if checks_p['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics_p['rms_ratio']:.6f}  "
          f"([{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks_p['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics_p['psnr_db']:.2f} dB  (>= {thr['psnr_db']} dB)  "
          f"{'OK' if checks_p['psnr_db'] else 'FAIL'}")

    print(f"\n  [p_final 2-D field]")
    print(f"  pearson_r : {metrics_pf['pearson_r']:.6f}  (>= {thr['pearson_r']})  "
          f"{'OK' if checks_pf['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics_pf['rms_ratio']:.6f}  "
          f"([{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks_pf['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics_pf['psnr_db']:.2f} dB  (>= {thr['psnr_db']} dB)  "
          f"{'OK' if checks_pf['psnr_db'] else 'FAIL'}")

    print(f"\n  runtime: k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    kw_aligned = {**kw, "pressure": kw_p_aligned}
    plot_comparison(kw_aligned, pkw_res, inputs, metrics_p, metrics_pf, status=status)

    header_lines = [
        "na_optimising_performance parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m  dy={DY:.6e} m",
        f"medium: c={C0} m/s  rho={RHO} kg/m3 (homogeneous)",
        f"source: EXAMPLE_source_two.bmp  p0_magnitude={P0_MAGNITUDE} Pa  smooth_p0=True",
        f"sensor: {NUM_SENSOR_POINTS}-pt Cartesian circle r={SENSOR_RADIUS:.4e} m "
        f"-> {n_sensors} grid pts",
        f"pml_size: {PML_SIZE}  pml_inside: True",
        f"nt={nt}  dt={dt:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
        "[sensor time-series]",
    ]
    report_lines = [
        f"pearson_r  = {metrics_p['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics_p['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics_p['psnr_db']:.2f} dB  (target >= {thr['psnr_db']} dB)",
        f"rmse       = {metrics_p['rmse']:.6e} Pa",
        f"peak_ratio = {metrics_p['peak_ratio']:.6f}",
        "",
        "[p_final 2-D field]",
        f"pearson_r  = {metrics_pf['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics_pf['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics_pf['psnr_db']:.2f} dB  (target >= {thr['psnr_db']} dB)",
        f"rmse       = {metrics_pf['rmse']:.6e} Pa",
        f"peak_ratio = {metrics_pf['peak_ratio']:.6f}",
    ]
    save_text_report(METRICS_PATH, "\n".join(header_lines), report_lines)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
