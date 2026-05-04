#!/usr/bin/env python3
"""
ivp_binary_sensor_mask_compare.py
===================================
Parity comparison for the upstream ``ivp_binary_sensor_mask`` example
(initial-value-problem with a binary arc sensor in a 2-D homogeneous medium).

Physical setup (matches k-wave-python ``examples/ivp_binary_sensor_mask.py``):
  Grid:    128×128  dx = dy = 0.1 mm
  Medium:  homogeneous  c = 1500 m/s
           absorption: alpha_coeff = 0.75 dB/(MHz^y cm)  alpha_power = 1.5
  Source:  initial pressure p0 (two discs: disc-1 at (50,50) r=8, 5 Pa;
           disc-2 at (80,60) r=5, 3 Pa)
  Sensor:  binary arc mask — make_circle centred at (64,64), radius=42,
           arc_angle = 3π/2 (three-quarter arc, ~93 grid points)
  PML:     default size, inside domain

Binary sensor mask ordering
---------------------------
k-wave-python indexes sensor rows by C-order linear index through the mask
(y-fastest for a (Nx, Ny) array).  pykwavers uses Fortran-order (x-fastest)
for a (Nx, Ny, 1) mask.  The standard C→Fortran permutation derived in
``ivp_homogeneous_medium_compare.py`` is applied before parity evaluation.

Outputs
-------
* ``output/ivp_binary_sensor_mask_compare.png``
* ``output/ivp_binary_sensor_mask_metrics.txt``

Usage
-----
  python examples/ivp_binary_sensor_mask_compare.py
  python examples/ivp_binary_sensor_mask_compare.py --no-cache
  python examples/ivp_binary_sensor_mask_compare.py --allow-failure
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
# Physical constants  (must match ivp_binary_sensor_mask.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 0.1e-3               # grid spacing [m]

C0          = 1500.0            # sound speed [m/s]
ALPHA_COEFF = 0.75              # absorption coefficient [dB/(MHz^y cm)]
ALPHA_POWER = 1.5               # absorption exponent

# Disc sources (1-indexed positions, matching MATLAB/k-wave convention)
DISC1_MAG = 5.0
DISC1_POS = [50, 50]
DISC1_R   = 8

DISC2_MAG = 3.0
DISC2_POS = [80, 60]
DISC2_R   = 5

# Binary arc sensor: centre (1-indexed), radius, arc angle
SENSOR_X_POS   = NX // 2       # = 64, 1-indexed
SENSOR_Y_POS   = NY // 2       # = 64, 1-indexed
SENSOR_RADIUS  = NX // 2 - 22  # = 42 grid points
SENSOR_ARC     = 3.0 * np.pi / 2.0   # 270° arc

PML_SIZE = 20

# ---------------------------------------------------------------------------
# Parity thresholds
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
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "ivp_binary_sensor_mask_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ivp_binary_sensor_mask_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "ivp_binary_sensor_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "ivp_binary_sensor_pykwavers_cache.npz"

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
            "n_sensors": int(d["n_sensors"]),
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
    """Construct grid, p0, binary arc sensor mask, and C→Fortran permutation.

    Both engines receive the same smoothed initial pressure and the same grid-
    aligned binary mask so that physical inputs are identical.
    """
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.utils.mapgen import make_circle, make_disc

    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    kgrid.makeTime(C0)
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    disc1 = DISC1_MAG * make_disc(Vector([NX, NY]), Vector(DISC1_POS), DISC1_R)
    disc2 = DISC2_MAG * make_disc(Vector([NX, NY]), Vector(DISC2_POS), DISC2_R)
    p0 = np.asarray(disc1 + disc2, dtype=np.float64)

    # Binary arc sensor mask (Nx, Ny) with integer values 0/1.
    sensor_mask_2d = np.asarray(
        make_circle(
            Vector([NX, NY]),
            Vector([SENSOR_X_POS, SENSOR_Y_POS]),
            SENSOR_RADIUS,
            SENSOR_ARC,
        ),
        dtype=bool,
    )
    n_sensors = int(sensor_mask_2d.sum())

    # C→Fortran permutation:
    # k-wave orders sensors by C-order scan of (Nx, Ny): y-fastest.
    # pykwavers orders by Fortran-order scan of (Nx, Ny, 1): x-fastest.
    # np.argwhere on a C-contiguous mask returns positions in C-order.
    # lexsort((x, y)) sorts primarily by y (last key), then by x — giving
    # the Fortran-order (x-fast) permutation of the C-order sequence.
    active_c = np.argwhere(sensor_mask_2d)          # (n_sensors, 2): [x, y]
    sensor_row_perm = np.lexsort(
        (active_c[:, 0], active_c[:, 1])
    )  # perm[j] = k-wave row for pykwavers row j

    return {
        "kgrid":           kgrid,
        "p0":              p0,
        "sensor_mask_2d":  sensor_mask_2d,
        "n_sensors":       n_sensors,
        "sensor_row_perm": sensor_row_perm,
        "nt":              nt,
        "dt":              dt,
    }


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(inputs: dict, *, no_cache: bool = False) -> dict:
    """Run k-wave-python with the binary arc sensor mask."""
    if not no_cache:
        cached = _load_cache(_KWAVE_CACHE)
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid          = inputs["kgrid"]
    p0             = inputs["p0"]
    sensor_mask_2d = inputs["sensor_mask_2d"]
    n_sensors      = inputs["n_sensors"]
    nt             = inputs["nt"]
    dt             = inputs["dt"]

    medium = kWaveMedium(
        sound_speed=C0,
        alpha_coeff=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )
    source = kSource()
    source.p0 = p0

    # Pass the binary (0/1) mask as integer array — k-wave-python expects bool/int.
    sensor = kSensor(mask=sensor_mask_2d.astype(np.uint8))
    sensor.record = ["p"]

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s, "
          f"n_sensors={n_sensors})...")
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
    if pressure.shape[0] != n_sensors:
        if pressure.shape[1] == n_sensors:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected k-wave sensor shape {pressure.shape}; "
                f"expected ({n_sensors}, {nt})"
            )

    _save_cache(_KWAVE_CACHE, pressure, nt, dt, runtime_s, n_sensors)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s,
            "n_sensors": n_sensors}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    """Run pykwavers with the binary arc sensor mask (expanded to Nx×Ny×1)."""
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    from kwave.utils.filters import smooth as kwave_smooth

    p0             = inputs["p0"]
    sensor_mask_2d = inputs["sensor_mask_2d"]
    n_sensors      = inputs["n_sensors"]
    nt             = inputs["nt"]
    dt             = inputs["dt"]

    grid = pkw.Grid(NX, NY, 1, DX, DY, DX)

    medium = pkw.Medium(
        sound_speed=C0,
        density=1000.0,
        alpha_coeff=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    # Apply same p0 smoothing as k-wave (smooth_p0=True default).
    p0_smooth = np.asarray(kwave_smooth(p0, restore_max=True), dtype=np.float64)
    p0_3d     = p0_smooth[:, :, None]
    source    = pkw.Source.from_initial_pressure(p0_3d)

    # Expand binary mask to (Nx, Ny, 1).
    sensor_mask_3d = sensor_mask_2d[:, :, None]
    sensor         = pkw.Sensor.from_mask(sensor_mask_3d)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running CPU PSTD  (Nt={nt}, dt={dt:.3e} s, "
          f"n_sensors={n_sensors})...")
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
                f"Unexpected pykwavers sensor shape {pressure.shape}; "
                f"expected ({n_sensors}, {nt})"
            )

    _save_cache(_PKWAV_CACHE, pressure, nt, dt, runtime_s, n_sensors)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s,
            "n_sensors": n_sensors}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    kw: dict,
    pkw_res: dict,
    sensor_mask_2d: np.ndarray,
    metrics: dict,
    *,
    status: str,
) -> None:
    kw_p  = kw["pressure"]    # (n_sensors, Nt) — after permutation
    py_p  = pkw_res["pressure"]
    diff  = py_p - kw_p

    vmax = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))
    dmax = float(max(np.abs(diff).max(), 1e-30))

    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # Top-left: initial sensor mask geometry
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(sensor_mask_2d.T, origin="lower", cmap="gray_r", aspect="equal")
    ax0.set_title("Binary Arc Sensor Mask")
    ax0.set_xlabel("x [grid pts]")
    ax0.set_ylabel("y [grid pts]")

    # Top-right + bottom-left: sensor data matrices
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(kw_p, aspect="auto", origin="lower", cmap="seismic",
                     vmin=-vmax, vmax=vmax)
    ax1.set_title("k-wave-python")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Sensor index")
    fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(py_p, aspect="auto", origin="lower", cmap="seismic",
                     vmin=-vmax, vmax=vmax)
    ax2.set_title("pykwavers")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Sensor index")
    fig.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1:])
    im3 = ax3.imshow(diff, aspect="auto", origin="lower", cmap="seismic",
                     vmin=-dmax, vmax=dmax)
    ax3.set_title("Difference (pykwavers − k-wave)")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Sensor index")
    fig.colorbar(im3, ax=ax3)

    fig.suptitle(
        f"ivp_binary_sensor_mask: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}  dx={DX*1e3:.2f} mm  c={C0} m/s  "
        f"arc_angle=3π/2  r={SENSOR_RADIUS} pts  "
        f"pearson_r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}",
        fontsize=9,
    )
    fig.savefig(str(FIGURE_PATH), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for ivp_binary_sensor_mask."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 70)
    print("ivp_binary_sensor_mask: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}   dx={DX*1e3:.3f} mm")
    print(f"  Medium : c={C0} m/s  alpha={ALPHA_COEFF} dB/(MHz^y cm)  "
          f"alpha_power={ALPHA_POWER}")
    print(f"  Source : two discs (IVP initial pressure)")
    print(f"  Sensor : binary arc  centre=({SENSOR_X_POS},{SENSOR_Y_POS})  "
          f"r={SENSOR_RADIUS}  arc=3π/2")
    print(f"  PML    : {PML_SIZE} pts inside")
    print("=" * 70)

    print("\n[0/2] Building shared inputs (grid, p0, binary sensor mask)...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    n_sensors = inputs["n_sensors"]
    print(f"  Nt={nt}  dt={dt:.3e} s  n_sensors={n_sensors}")

    # --- k-wave-python ---
    print("\n[1/2] k-wave-python (2-D PSTD, binary arc sensor)...")
    kw = run_kwave(inputs, no_cache=no_cache)
    kw_p = kw["pressure"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(kw_p**2))):.4e} Pa")

    # --- pykwavers ---
    print("\n[2/2] pykwavers (CPU PSTD, binary arc sensor expanded to Nx×Ny×1)...")
    pkw_res = run_pykwavers(inputs, no_cache=no_cache)
    py_p = pkw_res["pressure"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(py_p**2))):.4e} Pa")

    # --- Align sensor row ordering (C→Fortran permutation) ---
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
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  "
          f"pykwavers={pkw_res['runtime_s']:.1f}s")

    # --- Figure ---
    kw_aligned = {**kw, "pressure": kw_p_aligned}
    plot_comparison(
        kw_aligned,
        pkw_res,
        inputs["sensor_mask_2d"],
        metrics,
        status=status,
    )

    # --- Text report ---
    header_lines = [
        "ivp_binary_sensor_mask parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  alpha_coeff={ALPHA_COEFF}  alpha_power={ALPHA_POWER}",
        f"source: two discs (IVP)  smooth_p0=True",
        f"sensor: binary arc  centre=({SENSOR_X_POS},{SENSOR_Y_POS})  "
        f"radius={SENSOR_RADIUS}  arc=3pi/2  n_sensors={n_sensors}",
        f"pml_size: {PML_SIZE}  pml_inside: True",
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
