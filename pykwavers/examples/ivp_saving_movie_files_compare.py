#!/usr/bin/env python3
"""
ivp_saving_movie_files_compare.py
==================================
Parity comparison for the upstream ``ivp_saving_movie_files`` example
(initial-value-problem with two disc sources in a 2-D heterogeneous medium,
originally framed around movie-file output; physics parity is the target here).

Physical setup (matches k-wave-python ``examples/ivp_saving_movie_files.py``):
  Grid    : 128×128  dx = dy = 0.1 mm
  Medium  : heterogeneous
              c[:64, :]   = 1800 m/s  (first half, x-axis)
              c[64:, :]   = 1500 m/s  (second half)
              rho[:, :32] = 1000 kg/m³  (first quarter, y-axis)
              rho[:, 32:] = 1200 kg/m³  (last three-quarters)
  Source  : initial pressure p0 (two discs: disc-1 at (50,50) r=8, 5 Pa;
            disc-2 at (80,60) r=5, 3 Pa)
  Sensor  : centered Cartesian circle, r=4 mm, 50 points → grid mask
  Records : p (time-series) and p_final (end-state 2-D field)
  PML     : 20 grid points, inside domain

Density boundary differs from ``ivp_heterogeneous_medium`` by one cell:
  saving_movie_files: density[:, Ny//4:] = 1200  (from index 32)
  heterogeneous:      density[:, 31:]   = 1200  (from index 31)

Outputs
-------
* ``output/ivp_saving_movie_files_compare.png``  — sensor-data + p_final panels
* ``output/ivp_saving_movie_files_metrics.txt``

Usage
-----
  python examples/ivp_saving_movie_files_compare.py
  python examples/ivp_saving_movie_files_compare.py --no-cache
  python examples/ivp_saving_movie_files_compare.py --allow-failure
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
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical constants (must match ivp_saving_movie_files.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 0.1e-3  # [m]

C_FAST = 1800.0   # m/s — first half (x < Nx/2)
C_NOM  = 1500.0   # m/s — second half

RHO_LOW  = 1000.0   # kg/m³ — first quarter  (y < Ny//4 = 32)
RHO_HIGH = 1200.0   # kg/m³ — last 3/4       (y >= Ny//4 = 32)
RHO_BOUNDARY = NX // 4  # = 32; matches `density[:, Ny // 4 :]` in upstream

DISC1_MAG = 5.0
DISC1_POS = [50, 50]  # 1-indexed (k-wave convention inside make_disc)
DISC1_R   = 8

DISC2_MAG = 3.0
DISC2_POS = [80, 60]
DISC2_R   = 5

SENSOR_RADIUS     = 4e-3   # [m]
NUM_SENSOR_POINTS = 50

PML_SIZE = 20

# ---------------------------------------------------------------------------
# Parity thresholds
# Heterogeneous medium → reflections and wavefront distortion; both engines
# implement the same k-space PSTD scheme so high correlation is expected.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.90,
    "rms_ratio_min": 0.70,
    "rms_ratio_max": 1.35,
    "psnr_db":       15.0,
}

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "ivp_saving_movie_files_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ivp_saving_movie_files_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "ivp_saving_movie_files_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "ivp_saving_movie_files_pykwavers_cache.npz"

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
            "pressure":   np.asarray(d["pressure"],  dtype=np.float64),
            "p_final":    np.asarray(d["p_final"],   dtype=np.float64),
            "nt":         int(d["nt"]),
            "dt":         float(d["dt"]),
            "runtime_s":  float(d["runtime_s"]),
            "n_sensors":  int(d["n_sensors"]),
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
    """Construct grid, heterogeneous medium maps, p0, and grid sensor mask.

    Density boundary: rho[:, NY//4:] = RHO_HIGH, i.e. from index 32 onward.
    This matches the upstream script's ``density[:, Ny // 4 :] = 1200``.
    """
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.mapgen import make_cart_circle, make_disc
    from kwave.utils.conversion import cart2grid as kwave_cart2grid

    # Sound speed map
    c_kw = np.ones((NX, NY), dtype=np.float64) * C_NOM
    c_kw[:NX // 2, :] = C_FAST

    # Density map — boundary at Ny//4 = 32 (matches upstream `density[:, Ny // 4:]`)
    rho_kw = np.ones((NX, NY), dtype=np.float64) * RHO_LOW
    rho_kw[:, RHO_BOUNDARY:] = RHO_HIGH

    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    kgrid.makeTime(c_kw)  # uses max(c_kw) for CFL stability

    disc1 = DISC1_MAG * make_disc(Vector([NX, NY]), Vector(DISC1_POS), DISC1_R)
    disc2 = DISC2_MAG * make_disc(Vector([NX, NY]), Vector(DISC2_POS), DISC2_R)
    p0 = np.asarray(disc1 + disc2, dtype=np.float64)

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
        "c_kw":            c_kw,
        "rho_kw":          rho_kw,
        "p0":              p0,
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
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    from kwave.kmedium import kWaveMedium

    kgrid = inputs["kgrid"]
    nt    = inputs["nt"]
    dt    = inputs["dt"]
    n_sensors = inputs["n_sensors"]

    medium = kWaveMedium(sound_speed=inputs["c_kw"], density=inputs["rho_kw"])

    source   = kSource()
    source.p0 = inputs["p0"]  # k-wave applies smooth_p0=True by default

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

    from kwave.utils.filters import smooth as kwave_smooth

    nt        = inputs["nt"]
    dt        = inputs["dt"]
    n_sensors = inputs["n_sensors"]
    c_kw      = inputs["c_kw"]
    rho_kw    = inputs["rho_kw"]

    grid = pkw.Grid(NX, NY, 1, DX, DY, DX)

    c_3d   = c_kw[:, :, None].astype(np.float64)
    rho_3d = rho_kw[:, :, None].astype(np.float64)
    medium = pkw.Medium(sound_speed=c_3d, density=rho_3d)

    # Apply the same p0 smoothing k-wave applies (smooth_p0=True default)
    p0_smooth = np.asarray(kwave_smooth(inputs["p0"], restore_max=True), dtype=np.float64)
    p0_3d = p0_smooth[:, :, None]
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

    # p_final_field: full 3-D grid snapshot (NX, NY, 1) → extract 2-D (NX, NY)
    pf = result.p_final_field
    if pf is not None:
        p_final_3d = np.asarray(pf, dtype=np.float64)
        p_final = p_final_3d[:, :, 0] if p_final_3d.ndim == 3 else p_final_3d.squeeze()
    else:
        # Fall back to zeroes so the script still produces a valid comparison image
        p_final = np.zeros((NX, NY), dtype=np.float64)

    _save_cache(_PKWAV_CACHE, pressure, p_final, nt, dt, runtime_s, n_sensors)
    return {"pressure": pressure, "p_final": p_final,
            "nt": nt, "dt": dt, "runtime_s": runtime_s, "n_sensors": n_sensors}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    kw: dict, pkw_res: dict, metrics_p: dict, metrics_pf: dict, *, status: str
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

    # Row 0: sensor time-series
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

    # Row 1: p_final field
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

    fig.suptitle(
        f"ivp_saving_movie_files: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}  dx={DX*1e3:.2f} mm  "
        f"sensor r={metrics_p['pearson_r']:.4f}  "
        f"p_final r={metrics_pf['pearson_r']:.4f}",
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
        description="Compare pykwavers with k-wave-python for ivp_saving_movie_files."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 70)
    print("ivp_saving_movie_files: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}   dx={DX*1e3:.3f} mm")
    print(f"  Medium : c=[{C_NOM},{C_FAST}] m/s (heterogeneous by x-half)")
    print(f"           rho=[{RHO_LOW},{RHO_HIGH}] kg/m³ (boundary at y={RHO_BOUNDARY})")
    print(f"  Source : two discs (IVP initial pressure)")
    print(f"  Sensor : {NUM_SENSOR_POINTS}-pt Cartesian circle r={SENSOR_RADIUS*1e3:.1f} mm → grid mask")
    print(f"  Records: p (time-series) + p_final (2-D field)")
    print(f"  PML    : {PML_SIZE} pts inside")
    print("=" * 70)

    print("\n[0/2] Building shared inputs...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    n_sensors = inputs["n_sensors"]
    print(f"  Nt={nt}  dt={dt:.3e} s  n_sensors={n_sensors}")

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
    # Sensor time-series metrics
    metrics_p = compute_image_metrics(kw_p_aligned, pkw_res["pressure"])
    # p_final field metrics
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
    plot_comparison(kw_aligned, pkw_res, metrics_p, metrics_pf, status=status)

    header_lines = [
        "ivp_saving_movie_files parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c_fast={C_FAST} c_nom={C_NOM} m/s (heterogeneous, x-boundary at Nx/2={NX//2})",
        f"        rho_low={RHO_LOW} rho_high={RHO_HIGH} kg/m3 (y-boundary at {RHO_BOUNDARY})",
        f"source: two discs (IVP), smooth_p0=True",
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
