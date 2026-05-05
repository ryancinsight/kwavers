#!/usr/bin/env python3
"""
ivp_opposing_corners_sensor_mask_compare.py
============================================
Parity comparison for the upstream k-Wave example
``example_ivp_opposing_corners_sensor_mask.m``: an initial-value-problem in a
2-D homogeneous absorbing medium where the sensor mask is the union of two
rectangular regions specified by opposing-corner indices.

Physical setup (matches the MATLAB script verbatim)
---------------------------------------------------
Grid    : 128×128 with ``dx = dy = 0.1 mm``
Medium  : homogeneous water with absorption
          ``c = 1500 m/s, rho = 1000 kg/m³,
           alpha_coeff = 0.75 dB/(MHz^y cm), alpha_power = 1.5``
Source  : initial pressure ``p0 = disc1 + disc2``
          disc1: magnitude 5 Pa, centred at (50, 50), radius 8
          disc2: magnitude 3 Pa, centred at (80, 60), radius 5
          smoothed before propagation (matches k-wave default ``smooth_p0=True``)
Sensor  : union of two rectangles defined by opposing corners
          rect1: x ∈ [25, 30], y ∈ [31, 50]  →  6×20 =  120 grid points
          rect2: x ∈ [71, 80], y ∈ [81, 90]  → 10×10 =  100 grid points
          total: 220 binary sensor points
PML     : 20 grid points, inside domain

Comparison strategy
-------------------
The MATLAB original returns a structured output ``sensor_data(i).p`` per
rectangle. Both engines here consume a unified binary mask covering the union
of both rectangles, and the full ``(220 × Nt)`` sensor matrix is compared
under image-level Pearson r / RMS ratio / PSNR metrics. The sensor row order
between k-wave-python (C-order: x slowest, y fastest) and pykwavers
(Fortran-order: x fastest, y slowest) is reconciled with a precomputed
permutation, exactly as in ``ivp_homogeneous_medium_compare.py``.

Outputs
-------
* ``output/ivp_opposing_corners_sensor_mask_compare.png``  — 4-panel figure:
    1. p0 + sensor-mask overlay
    2. k-wave-python  sensor matrix (220 × Nt)
    3. pykwavers       sensor matrix (220 × Nt)
    4. difference (pykwavers − k-wave)
* ``output/ivp_opposing_corners_sensor_mask_metrics.txt``

Usage
-----
    python examples/ivp_opposing_corners_sensor_mask_compare.py
    python examples/ivp_opposing_corners_sensor_mask_compare.py --no-cache
    python examples/ivp_opposing_corners_sensor_mask_compare.py --allow-failure
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
# Physical constants (matches MATLAB example_ivp_opposing_corners_sensor_mask.m)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 0.1e-3  # grid spacing [m]

C0 = 1500.0  # sound speed [m/s]
RHO0 = 1000.0  # density [kg/m³]
ALPHA_COEFF = 0.75  # absorption coefficient [dB/(MHz^y cm)]
ALPHA_POWER = 1.5  # absorption power-law exponent

# Disc sources (1-indexed positions, MATLAB convention)
DISC1_MAG = 5.0
DISC1_POS = [50, 50]
DISC1_R = 8

DISC2_MAG = 3.0
DISC2_POS = [80, 60]
DISC2_R = 5

# Rectangular sensor regions defined by opposing corners (1-indexed inclusive)
RECT1 = (25, 31, 30, 50)  # (x_start, y_start, x_end, y_end)
RECT2 = (71, 81, 80, 90)

PML_SIZE = 20

# ---------------------------------------------------------------------------
# Parity thresholds — same expectations as ivp_homogeneous_medium_compare.py
# (same physics, only the sensor mask geometry changes).
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r": 0.97,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.25,
    "psnr_db": 20.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH = DEFAULT_OUTPUT_DIR / "ivp_opposing_corners_sensor_mask_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ivp_opposing_corners_sensor_mask_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "ivp_opposing_corners_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "ivp_opposing_corners_pykwavers_cache.npz"

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
            "pressure": np.asarray(d["pressure"], dtype=np.float64),
            "nt": int(d["nt"]),
            "dt": float(d["dt"]),
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
# Sensor-mask construction
# ---------------------------------------------------------------------------
def _rect_corners_to_mask(
    rect: tuple[int, int, int, int], nx: int, ny: int
) -> np.ndarray:
    """Return a (nx, ny) bool mask covering the inclusive 1-indexed rectangle.

    Convention (k-Wave):
        rect = (x_start, y_start, x_end, y_end), 1-indexed, both endpoints
        inclusive — exactly what k-wave's MATLAB ``sensor.mask`` corner-spec
        produces when the corners are passed to ``kspaceFirstOrder2D``.
    """
    x0, y0, x1, y1 = rect
    if not (1 <= x0 <= x1 <= nx and 1 <= y0 <= y1 <= ny):
        raise ValueError(f"Invalid rect {rect} for grid {nx}×{ny}")
    mask = np.zeros((nx, ny), dtype=bool)
    # 1-indexed inclusive → 0-indexed half-open
    mask[x0 - 1 : x1, y0 - 1 : y1] = True
    return mask


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
def build_shared_inputs() -> dict:
    """Construct grid, smoothed p0, and the rectangle-union binary sensor mask.

    Sensor row ordering: k-wave-python flattens with C-order (x slowest);
    pykwavers iterates Fortran-order (x fastest). The returned
    ``sensor_row_perm`` aligns kwave rows to pykwavers' row order via
    ``kwave_p[sensor_row_perm]``.
    """
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.filters import smooth
    from kwave.utils.mapgen import make_disc

    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    medium = kWaveMedium(
        sound_speed=C0, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER
    )
    kgrid.makeTime(medium.sound_speed)

    disc1 = DISC1_MAG * make_disc(Vector([NX, NY]), Vector(DISC1_POS), DISC1_R)
    disc2 = DISC2_MAG * make_disc(Vector([NX, NY]), Vector(DISC2_POS), DISC2_R)
    p0_raw = np.asarray(disc1 + disc2, dtype=np.float64)
    p0_smooth = np.asarray(smooth(p0_raw, restore_max=True), dtype=np.float64)

    # Build the rectangle-union binary sensor mask.
    rect1_mask = _rect_corners_to_mask(RECT1, NX, NY)
    rect2_mask = _rect_corners_to_mask(RECT2, NX, NY)
    sensor_mask_2d = rect1_mask | rect2_mask
    n_sensors = int(sensor_mask_2d.sum())

    # Sanity: rectangles are disjoint with the configured corners.
    overlap = int((rect1_mask & rect2_mask).sum())
    if overlap != 0:
        raise AssertionError(
            f"Rectangles must be disjoint; got {overlap} overlapping points"
        )
    expected = (
        (RECT1[2] - RECT1[0] + 1) * (RECT1[3] - RECT1[1] + 1)
        + (RECT2[2] - RECT2[0] + 1) * (RECT2[3] - RECT2[1] + 1)
    )
    if n_sensors != expected:
        raise AssertionError(
            f"Sensor count mismatch: got {n_sensors}, expected {expected}"
        )

    # C-order → Fortran-order row permutation (see ivp_homogeneous_medium_compare.py)
    active_c = np.argwhere(sensor_mask_2d)
    sensor_row_perm = np.lexsort((active_c[:, 0], active_c[:, 1]))

    return {
        "kgrid": kgrid,
        "medium": medium,
        "p0_smooth": p0_smooth,
        "sensor_mask_2d": sensor_mask_2d,
        "rect1_mask": rect1_mask,
        "rect2_mask": rect2_mask,
        "n_sensors": n_sensors,
        "sensor_row_perm": sensor_row_perm,
        "nt": int(kgrid.Nt),
        "dt": float(kgrid.dt),
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

    kgrid = inputs["kgrid"]
    medium = inputs["medium"]
    p0_smooth = inputs["p0_smooth"]
    sensor_mask_2d = inputs["sensor_mask_2d"]
    n_sensors = inputs["n_sensors"]
    nt = inputs["nt"]
    dt = inputs["dt"]

    source = kSource()
    source.p0 = p0_smooth

    sensor = kSensor(mask=sensor_mask_2d)
    sensor.record = ["p"]

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        smooth_p0=False,  # already smoothed in build_shared_inputs
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
                f"Unexpected k-wave sensor output shape {pressure.shape}; "
                f"expected ({n_sensors}, {nt})"
            )

    _save_cache(_KWAVE_CACHE, pressure, nt, dt, runtime_s, n_sensors)
    return {
        "pressure": pressure,
        "nt": nt,
        "dt": dt,
        "runtime_s": runtime_s,
        "n_sensors": n_sensors,
    }


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    p0_smooth = inputs["p0_smooth"]
    sensor_mask_2d = inputs["sensor_mask_2d"]
    n_sensors = inputs["n_sensors"]
    nt = inputs["nt"]
    dt = inputs["dt"]

    grid = pkw.Grid(NX, NY, 1, DX, DY, DX)
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        absorption=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    p0_3d = p0_smooth[:, :, None].astype(np.float64)
    source = pkw.Source.from_initial_pressure(p0_3d)

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
    if pressure.shape[0] != n_sensors:
        if pressure.shape[1] == n_sensors:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected pykwavers sensor output shape {pressure.shape}; "
                f"expected ({n_sensors}, {nt})"
            )

    _save_cache(_PKWAV_CACHE, pressure, nt, dt, runtime_s, n_sensors)
    return {
        "pressure": pressure,
        "nt": nt,
        "dt": dt,
        "runtime_s": runtime_s,
        "n_sensors": n_sensors,
    }


# ---------------------------------------------------------------------------
# Plotting (4 panels: geometry + 3 sensor matrices)
# ---------------------------------------------------------------------------
def plot_comparison(
    inputs: dict, kw: dict, pkw_res: dict, metrics: dict, *, status: str
) -> None:
    p0 = inputs["p0_smooth"]
    rect1_mask = inputs["rect1_mask"]
    rect2_mask = inputs["rect2_mask"]
    kw_p = kw["pressure"]
    py_p = pkw_res["pressure"]
    diff = py_p - kw_p

    vmax = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))
    dmax = float(max(np.abs(diff).max(), 1e-30))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    # (0,:) Geometry — p0 with sensor rectangle overlay
    ax_geom = fig.add_subplot(gs[0, 0])
    p0_max = float(np.abs(p0).max())
    im_g = ax_geom.imshow(
        p0.T,
        origin="lower",
        cmap="seismic",
        vmin=-p0_max,
        vmax=p0_max,
        extent=[0, NX, 0, NY],
    )
    # Outline rectangles
    overlay = np.zeros((NX, NY, 4), dtype=np.float32)
    overlay[rect1_mask, 0] = 1.0  # red
    overlay[rect1_mask, 3] = 0.45
    overlay[rect2_mask, 2] = 1.0  # blue
    overlay[rect2_mask, 3] = 0.45
    ax_geom.imshow(np.transpose(overlay, (1, 0, 2)), origin="lower", extent=[0, NX, 0, NY])
    ax_geom.set_title(
        f"p0 (smoothed) + sensor rects\n"
        f"rect1 (red) {RECT1}  rect2 (blue) {RECT2}",
        fontsize=9,
    )
    ax_geom.set_xlabel("x [grid]")
    ax_geom.set_ylabel("y [grid]")
    fig.colorbar(im_g, ax=ax_geom, fraction=0.046, pad=0.04)

    # (0, 1:3) Side-by-side sensor matrices stacked in their row
    ax_kw = fig.add_subplot(gs[0, 1])
    im_kw = ax_kw.imshow(
        kw_p, aspect="auto", origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax
    )
    ax_kw.set_title("k-wave-python sensor matrix")
    ax_kw.set_xlabel("Time step")
    ax_kw.set_ylabel("Sensor index")
    fig.colorbar(im_kw, ax=ax_kw, fraction=0.046, pad=0.04)

    ax_py = fig.add_subplot(gs[0, 2])
    im_py = ax_py.imshow(
        py_p, aspect="auto", origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax
    )
    ax_py.set_title("pykwavers sensor matrix")
    ax_py.set_xlabel("Time step")
    ax_py.set_ylabel("Sensor index")
    fig.colorbar(im_py, ax=ax_py, fraction=0.046, pad=0.04)

    # Bottom row: difference (full-width) + a representative time trace per rectangle
    ax_diff = fig.add_subplot(gs[1, 0:2])
    im_diff = ax_diff.imshow(
        diff, aspect="auto", origin="lower", cmap="seismic", vmin=-dmax, vmax=dmax
    )
    ax_diff.set_title(
        f"difference (pykwavers − k-wave)   max|Δ| = {dmax:.3e} Pa"
    )
    ax_diff.set_xlabel("Time step")
    ax_diff.set_ylabel("Sensor index")
    fig.colorbar(im_diff, ax=ax_diff, fraction=0.025, pad=0.04)

    # Trace through the centre point of each rectangle (post-permutation row)
    # Find the indices in the unified mask that correspond to each rect's centre.
    perm = inputs["sensor_row_perm"]
    flat_active = np.argwhere(inputs["sensor_mask_2d"])  # C-order
    flat_active_p = flat_active[perm]  # pykwavers row order

    def _row_for_centre(rect: tuple[int, int, int, int]) -> int:
        cx = (rect[0] + rect[2]) // 2 - 1  # 0-indexed
        cy = (rect[1] + rect[3]) // 2 - 1
        diffs = np.abs(flat_active_p[:, 0] - cx) + np.abs(flat_active_p[:, 1] - cy)
        return int(np.argmin(diffs))

    row1 = _row_for_centre(RECT1)
    row2 = _row_for_centre(RECT2)

    ax_tr = fig.add_subplot(gs[1, 2])
    t_axis = np.arange(kw_p.shape[1])
    ax_tr.plot(t_axis, kw_p[row1], "r--", alpha=0.85, lw=1.4, label="k-wave rect1")
    ax_tr.plot(
        t_axis, py_p[row1], "r-", alpha=0.85, lw=1.0, label="pykwavers rect1"
    )
    ax_tr.plot(t_axis, kw_p[row2], "b--", alpha=0.85, lw=1.4, label="k-wave rect2")
    ax_tr.plot(
        t_axis, py_p[row2], "b-", alpha=0.85, lw=1.0, label="pykwavers rect2"
    )
    ax_tr.set_title("traces at rect-centres")
    ax_tr.set_xlabel("Time step")
    ax_tr.set_ylabel("Pressure [Pa]")
    ax_tr.legend(fontsize=7, loc="best")
    ax_tr.grid(True, alpha=0.3)

    fig.suptitle(
        f"ivp_opposing_corners_sensor_mask: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}  dx={DX*1e3:.2f} mm  "
        f"sensor: {kw_p.shape[0]} pts (rect1∪rect2)  "
        f"r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}  "
        f"PSNR={metrics['psnr_db']:.1f} dB",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(str(FIGURE_PATH), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pykwavers with k-wave-python for "
            "ivp_opposing_corners_sensor_mask."
        )
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force a fresh run (ignore cached NPZ files).",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Exit 0 even when parity targets fail.",
    )
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 72)
    print("ivp_opposing_corners_sensor_mask: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}   dx={DX*1e3:.3f} mm")
    print(
        f"  Medium : c={C0} m/s  rho={RHO0} kg/m³  "
        f"alpha={ALPHA_COEFF} dB/(MHz^y cm)  y={ALPHA_POWER}"
    )
    print("  Source : two discs (IVP initial pressure), smoothed")
    print(f"  Sensor : rect1 {RECT1} ∪ rect2 {RECT2} (binary mask)")
    print(f"  PML    : {PML_SIZE} pts inside")
    print("=" * 72)

    print("\n[0/2] Building shared inputs (grid, smooth p0, sensor mask)...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    n_sensors = inputs["n_sensors"]
    print(f"  Nt={nt}  dt={dt:.3e} s  n_sensors={n_sensors}")

    print("\n[1/2] k-wave-python (kspaceFirstOrder, 2-D PSTD)...")
    kw = run_kwave(inputs, no_cache=no_cache)
    kw_p = kw["pressure"]
    print(
        f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa  "
        f"rms={float(np.sqrt(np.mean(kw_p**2))):.4e} Pa"
    )

    print("\n[2/2] pykwavers (CPU PSTD, Source.from_initial_pressure)...")
    pkw_res = run_pykwavers(inputs, no_cache=no_cache)
    py_p = pkw_res["pressure"]
    print(
        f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa  "
        f"rms={float(np.sqrt(np.mean(py_p**2))):.4e} Pa"
    )

    sensor_row_perm = inputs["sensor_row_perm"]
    kw_p_aligned = kw_p[sensor_row_perm]

    print("\n--- Parity evaluation ---")
    metrics = compute_image_metrics(kw_p_aligned, py_p)

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"] >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"]
        <= metrics["rms_ratio"]
        <= thr["rms_ratio_max"],
        "psnr_db": metrics["psnr_db"] >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status    : {status}")
    print(
        f"  pearson_r : {metrics['pearson_r']:.6f}  "
        f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}"
    )
    print(
        f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
        f"{'OK' if checks['rms_ratio'] else 'FAIL'}"
    )
    print(
        f"  psnr_db   : {metrics['psnr_db']:.2f} dB  "
        f"(target >= {thr['psnr_db']} dB)  {'OK' if checks['psnr_db'] else 'FAIL'}"
    )
    print(f"  rmse      : {metrics['rmse']:.4e} Pa")
    print(
        f"  runtime   : k-wave={kw['runtime_s']:.1f}s  "
        f"pykwavers={pkw_res['runtime_s']:.1f}s"
    )

    kw_aligned = {**kw, "pressure": kw_p_aligned}
    plot_comparison(inputs, kw_aligned, pkw_res, metrics, status=status)

    header_lines = [
        "ivp_opposing_corners_sensor_mask parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  rho={RHO0} kg/m3  "
        f"alpha_coeff={ALPHA_COEFF}  alpha_power={ALPHA_POWER}",
        "source: two discs (IVP), smoothed (restore_max=True)",
        f"sensor: rect1{RECT1} ∪ rect2{RECT2} → {n_sensors} unique grid points",
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
