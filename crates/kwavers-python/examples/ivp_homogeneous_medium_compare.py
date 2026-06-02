#!/usr/bin/env python3
"""
ivp_homogeneous_medium_compare.py
==================================
Parity comparison for the upstream ``ivp_homogeneous_medium`` example
(initial-value-problem with two disc sources in a 2-D homogeneous absorbing medium).

Physical setup (matches k-wave-python ``examples/ivp_homogeneous_medium.py``):
  Grid:    128×128  dx = dy = 0.1 mm
  Medium:  homogeneous  c = 1500 m/s  rho = 1000 kg/m³
           absorption: alpha_coeff = 0.75 dB/(MHz^y cm)  alpha_power = 1.5
  Source:  initial pressure p0 (two discs: disc-1 at (50,50) r=8, 5 Pa;
           disc-2 at (80,60) r=5, 3 Pa)  smoothed before simulation
  Sensor:  centered Cartesian circle, r=4 mm, 50 points, converted to a
           C-order grid mask (50 unique grid points)
  PML:     20 grid points, inside domain

Comparison strategy
-------------------
Both engines receive the same smooth-filtered initial-pressure field and the
same C-order grid sensor mask so that sensor row ordering is identical.
The full (50 × Nt) sensor matrix is compared with image-level Pearson r and
RMS ratio metrics.

Outputs
-------
* ``output/ivp_homogeneous_medium_compare.png``
* ``output/ivp_homogeneous_medium_metrics.txt``

Usage
-----
  python examples/ivp_homogeneous_medium_compare.py
  python examples/ivp_homogeneous_medium_compare.py --no-cache
  python examples/ivp_homogeneous_medium_compare.py --allow-failure
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
# Physical constants  (must match ivp_homogeneous_medium.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 0.1e-3               # grid spacing [m]

C0          = 1500.0            # sound speed [m/s]
RHO0        = 1000.0            # density [kg/m³]
ALPHA_COEFF = 0.75              # absorption coefficient [dB/(MHz^y cm)]
ALPHA_POWER = 1.5               # absorption power law exponent

# Disc sources (1-indexed positions, matching MATLAB/k-wave convention)
DISC1_MAG = 5.0
DISC1_POS = [50, 50]            # 1-indexed
DISC1_R   = 8

DISC2_MAG = 3.0
DISC2_POS = [80, 60]            # 1-indexed
DISC2_R   = 5

SENSOR_RADIUS      = 4e-3       # [m]
NUM_SENSOR_POINTS  = 50

PML_SIZE           = 20         # grid points

# ---------------------------------------------------------------------------
# Parity thresholds
# IVP with absorption: initial wavefront expands through a lossy medium.
# Both PSTD solvers use the same k-space formulation so high correlation
# is expected for the full sensor matrix.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.97,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.25,
    "psnr_db":       20.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "ivp_homogeneous_medium_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ivp_homogeneous_medium_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "ivp_homogeneous_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "ivp_homogeneous_pykwavers_cache.npz"

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
            "pressure":   np.asarray(d["pressure"],   dtype=np.float64),
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
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
        n_sensors=np.array(n_sensors, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
def build_shared_inputs() -> dict:
    """Construct grid, smoothed p0, and grid-based sensor mask.

    Uses k-wave-python utilities to build the shared inputs so that the
    exact same disc geometry, smoothing kernel, and Cartesian→grid mapping
    are applied to both engines.

    Sensor row ordering
    -------------------
    k-wave-python's ``kspaceFirstOrder`` extracts sensor rows in **C-order**
    (x varies slowest, y varies fastest) via ``mask.ravel()[idx]``.
    pykwavers' ``SensorRecorder`` iterates in **Fortran-order** (x varies
    fastest, y varies slowest).  The returned ``sensor_row_perm`` permutation
    satisfies: ``kwave_p[sensor_row_perm]`` is in pykwavers' row order.
    """
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.filters import smooth
    from kwave.utils.mapgen import make_cart_circle, make_disc
    from kwave.utils.conversion import cart2grid as kwave_cart2grid

    kgrid  = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    medium = kWaveMedium(sound_speed=C0, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER)
    kgrid.makeTime(medium.sound_speed)

    disc1 = DISC1_MAG * make_disc(Vector([NX, NY]), Vector(DISC1_POS), DISC1_R)
    disc2 = DISC2_MAG * make_disc(Vector([NX, NY]), Vector(DISC2_POS), DISC2_R)
    p0_raw = np.asarray(disc1 + disc2, dtype=np.float64)

    # Apply the same smoothing that kspaceFirstOrder applies by default (smooth_p0=True)
    p0_smooth = np.asarray(smooth(p0_raw, restore_max=True), dtype=np.float64)

    # Cartesian circle → C-order grid mask (both engines use the same mask)
    sensor_circle = make_cart_circle(SENSOR_RADIUS, NUM_SENSOR_POINTS)
    sensor_mask_2d, _, _ = kwave_cart2grid(kgrid, sensor_circle, order="C")
    sensor_mask_2d = np.asarray(sensor_mask_2d, dtype=bool)
    n_sensors = int(sensor_mask_2d.sum())

    # Build C-order → Fortran-order permutation for sensor rows.
    # active_c[i] = (x, y) of k-wave row i (C-order: primary x, secondary y).
    # pykwavers row j corresponds to active_c[active_f_idx[j]] (Fortran-order:
    # primary y, secondary x).  So kwave_p[active_f_idx] aligns with pykwavers.
    active_c = np.argwhere(sensor_mask_2d)          # (n, 2), C-order
    sensor_row_perm = np.lexsort((active_c[:, 0], active_c[:, 1]))  # F-order indices into active_c

    return {
        "kgrid":            kgrid,
        "medium":           medium,
        "p0_smooth":        p0_smooth,
        "sensor_mask_2d":   sensor_mask_2d,
        "n_sensors":        n_sensors,
        "sensor_row_perm":  sensor_row_perm,
        "nt":               int(kgrid.Nt),
        "dt":               float(kgrid.dt),
    }


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(inputs: dict, *, no_cache: bool = False) -> dict:
    """Run k-wave-python with the smoothed p0 and grid sensor mask.

    ``smooth_p0=False`` is passed because the source is already smoothed in
    ``build_shared_inputs``; this prevents double-smoothing.
    """
    if not no_cache:
        cached = _load_cache(_KWAVE_CACHE)
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid      = inputs["kgrid"]
    medium     = inputs["medium"]
    p0_smooth  = inputs["p0_smooth"]
    sensor_mask_2d = inputs["sensor_mask_2d"]  # (128, 128) bool
    n_sensors  = inputs["n_sensors"]
    nt         = inputs["nt"]
    dt         = inputs["dt"]

    source       = kSource()
    source.p0    = p0_smooth

    sensor       = kSensor(mask=sensor_mask_2d)
    sensor.record = ["p"]

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        smooth_p0=False,          # already smoothed
        pml_inside=True,
        pml_size=PML_SIZE,
        backend="python",
        device="cpu",
        quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result["p"], dtype=np.float64)  # (n_sensors, Nt)
    if pressure.shape[0] != n_sensors:
        # May come transposed
        if pressure.shape[1] == n_sensors:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected k-wave sensor output shape {pressure.shape}; "
                f"expected ({n_sensors}, {nt})"
            )

    _save_cache(_KWAVE_CACHE, pressure, nt, dt, runtime_s, n_sensors)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s, "n_sensors": n_sensors}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    """Run pykwavers with Source.from_initial_pressure and a grid sensor mask."""
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    p0_smooth      = inputs["p0_smooth"]
    sensor_mask_2d = inputs["sensor_mask_2d"]  # (128, 128) bool
    n_sensors      = inputs["n_sensors"]
    nt             = inputs["nt"]
    dt             = inputs["dt"]

    # pykwavers is quasi-2D: nz = 1
    grid   = pkw.Grid(NX, NY, 1, DX, DY, DX)
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        absorption=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    # Expand p0 to (NX, NY, 1)
    p0_3d = p0_smooth[:, :, None].astype(np.float64)
    source = pkw.Source.from_initial_pressure(p0_3d)

    # Expand sensor mask to (NX, NY, 1) bool
    sensor_mask_3d = sensor_mask_2d[:, :, None]
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running CPU PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64)
    # Normalize to (n_sensors, Nt)
    if pressure.shape[0] != n_sensors:
        if pressure.shape[1] == n_sensors:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected pykwavers sensor output shape {pressure.shape}; "
                f"expected ({n_sensors}, {nt})"
            )

    _save_cache(_PKWAV_CACHE, pressure, nt, dt, runtime_s, n_sensors)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s, "n_sensors": n_sensors}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict, metrics: dict, *, status: str) -> None:
    """Three-panel: k-wave sensor image, pykwavers sensor image, difference."""
    kw_p  = kw["pressure"]
    py_p  = pkw_res["pressure"]
    diff  = py_p - kw_p

    vmax = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))
    dmax = float(max(np.abs(diff).max(), 1e-30))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(kw_p,  aspect="auto", origin="lower", cmap="seismic",
                         vmin=-vmax, vmax=vmax)
    axes[0].set_title("k-wave-python")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Sensor index")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(py_p,  aspect="auto", origin="lower", cmap="seismic",
                         vmin=-vmax, vmax=vmax)
    axes[1].set_title("pykwavers")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Sensor index")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="seismic",
                         vmin=-dmax, vmax=dmax)
    axes[2].set_title("difference (pykwavers − k-wave)")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Sensor index")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(
        f"ivp_homogeneous_medium: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}  dx={DX*1e3:.2f} mm  50-pt circular sensor  "
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
        description="Compare pykwavers with k-wave-python for ivp_homogeneous_medium."
    )
    parser.add_argument("--no-cache",      action="store_true",
                        help="Force a fresh run (ignore cached NPZ files).")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 70)
    print("ivp_homogeneous_medium: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}   dx={DX*1e3:.3f} mm")
    print(f"  Medium : c={C0} m/s  rho={RHO0} kg/m³  "
          f"alpha={ALPHA_COEFF} dB/(MHz^y cm)  y={ALPHA_POWER}")
    print(f"  Source : two discs (IVP initial pressure), smoothed")
    print(f"  Sensor : {NUM_SENSOR_POINTS}-pt Cartesian circle r={SENSOR_RADIUS*1e3:.1f} mm → grid mask")
    print(f"  PML    : {PML_SIZE} pts inside")
    print("=" * 70)

    print("\n[0/2] Building shared inputs (grid, smooth p0, sensor mask)...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    n_sensors = inputs["n_sensors"]
    print(f"  Nt={nt}  dt={dt:.3e} s  n_sensors={n_sensors}")

    # --- k-wave-python ---
    print("\n[1/2] k-wave-python (new kspaceFirstOrder API, 2-D PSTD)...")
    kw = run_kwave(inputs, no_cache=no_cache)
    kw_p = kw["pressure"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(kw_p**2))):.4e} Pa")

    # --- pykwavers ---
    print("\n[2/2] pykwavers (CPU PSTD, Source.from_initial_pressure)...")
    pkw_res = run_pykwavers(inputs, no_cache=no_cache)
    py_p = pkw_res["pressure"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(py_p**2))):.4e} Pa")

    # --- Align sensor row ordering ---
    # k-wave-python uses C-order traversal; pykwavers uses Fortran-order.
    # Apply the precomputed permutation: kw_p[perm] is in pykwavers' row order.
    sensor_row_perm = inputs["sensor_row_perm"]
    kw_p_aligned = kw_p[sensor_row_perm]

    # --- Parity ---
    print("\n--- Parity evaluation ---")
    metrics = compute_image_metrics(kw_p_aligned, py_p)

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics["psnr_db"]   >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics['psnr_db']:.2f} dB  "
          f"(target >= {thr['psnr_db']} dB)  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  rmse      : {metrics['rmse']:.4e} Pa")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    # --- Figure ---
    kw_aligned = {**kw, "pressure": kw_p_aligned}
    plot_comparison(kw_aligned, pkw_res, metrics, status=status)

    # --- Text report ---
    header_lines = [
        "ivp_homogeneous_medium parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  rho={RHO0} kg/m3  "
        f"alpha_coeff={ALPHA_COEFF}  alpha_power={ALPHA_POWER}",
        f"source: two discs (IVP), smoothed (restore_max=True)",
        f"sensor: {NUM_SENSOR_POINTS}-pt Cartesian circle r={SENSOR_RADIUS:.4e} m → "
        f"{n_sensors} unique C-order grid points",
        f"pml_size: {PML_SIZE}  pml_inside: True  smooth_p0: False (pre-smoothed)",
        f"nt={nt}  dt={dt:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ]
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f} dB  (target >= {thr['psnr_db']} dB)",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw_p_aligned).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(py_p).max()):.6e}",
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
