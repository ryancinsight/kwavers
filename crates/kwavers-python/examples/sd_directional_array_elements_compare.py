#!/usr/bin/env python3
"""
sd_directional_array_elements_compare.py
==========================================
Parity comparison for the upstream ``sd_directional_array_elements`` example
(curved 13-element semicircular array driven by a plane-wave source).

Physical setup (matches k-wave-python ``examples/sd_directional_array_elements.py``):
  Grid:    180×180   dx = dy = 0.1 mm
  Medium:  homogeneous, c = 1500 m/s, lossless
  Time:    t_end = 12 μs (via makeTime)
  Source:  plane-wave — entire row 139 (0-based), 1 MHz sinusoid filtered to
           grid Nyquist.  k-wave broadcasts a 1D signal to all Ny=180 active
           source points; pykwavers does the same via Source.from_mask
           with a 1D signal.
  Sensor:  semicircular arc (radius 65 grid pts, arc = π, centred on grid),
           divided into Ne = 13 elements with 2-point gaps.  Each element
           records time-series at several grid points, then spatially averages
           to one trace per element.
  PML:     inside; pml_alpha = (2, 0) — active in x, disabled in y.
           Disabling y-PML makes the plane-wave source appear infinitely wide.
           Replicated in pykwavers via sim.set_pml_alpha_xyz(2.0, 0.0, 2.0).

Sensor row ordering and element assignment
------------------------------------------
k-wave records sensor rows in C-order (y changes fastest for a (Nx, Ny) mask).
pykwavers records in Fortran-order (x changes fastest).

The C→Fortran permutation perm = np.lexsort((active[:,0], active[:,1])) maps
C-order row index i → Fortran position perm⁻¹[i].
For each element e, k-wave rows data_rows_kw[e] are re-mapped to pykwavers rows
data_rows_py[e] = inv_perm[data_rows_kw[e]] using the inverse permutation.
Both engines' per-element averages are then computed independently and compared.

Outputs
-------
* ``output/sd_directional_array_elements_compare.png``
* ``output/sd_directional_array_elements_metrics.txt``

Usage
-----
  python examples/sd_directional_array_elements_compare.py
  python examples/sd_directional_array_elements_compare.py --no-cache
  python examples/sd_directional_array_elements_compare.py --allow-failure
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
# Physical constants  (must match sd_directional_array_elements.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 180
DX = DY = 0.1e-3               # [m]
C0 = 1500.0                     # [m/s]

T_END       = 12e-6             # [s] — via kgrid.makeTime(..., t_end=T_END)
SOURCE_ROW  = 139               # 0-based x-index of plane-wave source
SOURCE_FREQ = 1e6               # [Hz]
SOURCE_MAG  = 0.5               # [Pa]

SEMI_RADIUS = 65                # [grid points]
SEMI_ARC    = np.pi             # half-circle
NE          = 13                # number of elements

F_MAX = C0 / (2.0 * DX)        # = 7.5 MHz — metadata for Source.from_mask

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
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "sd_directional_array_elements_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "sd_directional_array_elements_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "sd_directional_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "sd_directional_pykwavers_cache.npz"

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
            "p_raw":     np.asarray(d["p_raw"],     dtype=np.float64),
            "nt":        int(d["nt"]),
            "dt":        float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_cache(path: os.PathLike, p_raw: np.ndarray, nt: int,
                dt: float, runtime_s: float) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        p_raw=np.asarray(p_raw, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Shared inputs — grid, medium, filtered source, sensor geometry
# ---------------------------------------------------------------------------
def build_shared_inputs() -> dict:
    """Build shared grid, source signal, sensor mask, element assignments, and permutation."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.filters import filter_time_series
    from kwave.utils.mapgen import make_circle

    kgrid  = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    medium = kWaveMedium(sound_speed=C0)
    kgrid.makeTime(medium.sound_speed, t_end=T_END)

    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    t_array = np.asarray(kgrid.t_array).ravel()
    sig_raw = SOURCE_MAG * np.sin(2 * np.pi * SOURCE_FREQ * t_array)
    # filter_time_series returns (1, Nt); ravel to 1D
    filtered = np.asarray(
        filter_time_series(kgrid, medium, sig_raw.reshape(1, -1)),
        dtype=np.float64,
    ).ravel()

    # Source mask: full row 139
    src_mask_2d = np.zeros((NX, NY), dtype=np.float64)
    src_mask_2d[SOURCE_ROW, :] = 1.0

    # Semicircular arc sensor (radius=65, arc=π, centre on grid)
    arc = np.asarray(
        make_circle(Vector([NX, NY]), Vector([NX // 2, NY // 2]),
                    SEMI_RADIUS, SEMI_ARC),
        dtype=np.float64,
    )
    arc_flat_indices = np.flatnonzero(arc)   # C-order flat indices, sorted ascending
    Nv = len(arc_flat_indices)

    # Sort arc points by angle (matches original sd_directional_array_elements.py)
    kgrid_tmp = kWaveGrid(Vector([NX, NY]), Vector([DX, DX]))
    x_vec = np.asarray(kgrid_tmp.x_vec).ravel()
    y_vec = np.asarray(kgrid_tmp.y_vec).ravel()
    rows, cols = np.unravel_index(arc_flat_indices, (NX, NY))
    xp = x_vec[rows]
    yp = y_vec[cols]
    arc_angles = np.where(
        xp != 0,
        np.arctan(yp / np.where(xp != 0, xp, 1.0)),
        np.sign(yp) * np.pi / 2,
    )
    sorted_order = np.argsort(arc_angles)
    sorted_arc_flat = arc_flat_indices[sorted_order]

    # Divide into NE elements with 2-point gaps
    sensor_mask_2d = np.zeros((NX, NY), dtype=np.float64)
    element_kw_rows: list[np.ndarray] = []   # k-wave C-order rows per element

    sensor_flat_indices: np.ndarray | None = None   # will be set after mask is built

    # First pass: collect element voxel flat indices and fill sensor_mask_2d
    element_flat_indices: list[np.ndarray] = []
    for loop in range(NE):
        start = int(np.floor(loop * Nv / NE)) + 1
        end_  = int(np.floor((loop + 1) * Nv / NE)) - 1
        voxel_flat = sorted_arc_flat[start:end_]
        element_flat_indices.append(voxel_flat)
        sensor_mask_2d.flat[voxel_flat] = 1.0

    n_active = int(sensor_mask_2d.sum())

    # Compute C-order row indices for each element (searchsorted over sorted active flat indices)
    sensor_flat_all = np.sort(np.flatnonzero(sensor_mask_2d))  # ascending C-order
    for loop in range(NE):
        data_rows = np.searchsorted(sensor_flat_all, element_flat_indices[loop])
        element_kw_rows.append(data_rows)

    # C→Fortran permutation for the sensor mask
    active_c = np.argwhere(sensor_mask_2d.astype(bool))   # shape (n_active, 2): [x, y]
    perm     = np.lexsort((active_c[:, 0], active_c[:, 1]))
    inv_perm = np.argsort(perm)

    # pykwavers rows for each element: inv_perm[kw_row] → Fortran row
    element_py_rows: list[np.ndarray] = []
    for kw_rows in element_kw_rows:
        element_py_rows.append(inv_perm[kw_rows])

    return {
        "kgrid":            kgrid,
        "medium":           medium,
        "filtered_sig":     filtered,
        "src_mask_2d":      src_mask_2d,
        "sensor_mask_2d":   sensor_mask_2d,
        "n_active":         n_active,
        "perm":             perm,
        "element_kw_rows":  element_kw_rows,
        "element_py_rows":  element_py_rows,
        "nt":               nt,
        "dt":               dt,
    }


# ---------------------------------------------------------------------------
# k-wave-python run — returns raw sensor data (n_active, Nt) in C-order
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

    kgrid    = inputs["kgrid"]
    medium   = inputs["medium"]
    nt       = inputs["nt"]
    dt       = inputs["dt"]
    n_active = inputs["n_active"]

    source        = kSource()
    source.p_mask = inputs["src_mask_2d"]
    source.p      = inputs["filtered_sig"].reshape(1, -1)   # (1, Nt)

    sensor = kSensor(mask=inputs["sensor_mask_2d"].astype(bool), record=["p"])

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s, n_sens={n_active})...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        pml_inside=True,
        pml_alpha=(2, 0),        # active in x, disabled in y
        backend="python",
        device="cpu",
        quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    p_raw = np.asarray(result["p"], dtype=np.float64)
    if p_raw.shape[0] != n_active:
        if p_raw.shape[1] == n_active:
            p_raw = p_raw.T
        else:
            raise AssertionError(
                f"Unexpected k-wave shape {p_raw.shape}; expected ({n_active}, {nt})"
            )

    _save_cache(_KWAVE_CACHE, p_raw, nt, dt, runtime_s)
    return {"p_raw": p_raw, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# pykwavers run — returns raw sensor data (n_active, Nt) in Fortran-order
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    nt       = inputs["nt"]
    dt       = inputs["dt"]
    n_active = inputs["n_active"]

    grid   = pkw.Grid(NX, NY, 1, DX, DY, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=1000.0)

    src_mask_3d = inputs["src_mask_2d"][:, :, None]
    filtered    = inputs["filtered_sig"]    # 1D — broadcast to all NY source pts
    source = pkw.Source.from_mask(src_mask_3d, filtered, F_MAX)

    sensor_mask_3d = inputs["sensor_mask_2d"][:, :, None].astype(bool)
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)
    # Replicate k-wave's pml_alpha=(2,0): PML active in x, disabled in y
    sim.set_pml_alpha_xyz(2.0, 0.0, 2.0)

    print(f"  [pykwavers] Running quasi-2D PSTD  (Nt={nt}, dt={dt:.3e} s, n_sens={n_active})...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    p_raw = np.asarray(result.sensor_data, dtype=np.float64)
    if p_raw.shape[0] != n_active:
        if p_raw.shape[1] == n_active:
            p_raw = p_raw.T
        else:
            raise AssertionError(
                f"Unexpected pykwavers shape {p_raw.shape}; expected ({n_active}, {nt})"
            )

    _save_cache(_PKWAV_CACHE, p_raw, nt, dt, runtime_s)
    return {"p_raw": p_raw, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Per-element averaging
# ---------------------------------------------------------------------------
def compute_element_data(p_raw: np.ndarray,
                         element_rows: list[np.ndarray],
                         nt: int) -> np.ndarray:
    """Average sensor rows for each of the NE elements.

    Parameters
    ----------
    p_raw        : (n_active, Nt) sensor matrix
    element_rows : list[ndarray] — row indices into p_raw per element
    nt           : number of time steps

    Returns
    -------
    element_data : (NE, Nt) per-element averaged traces
    """
    element_data = np.zeros((NE, nt), dtype=np.float64)
    for e, rows in enumerate(element_rows):
        element_data[e, :] = p_raw[rows, :].mean(axis=0)
    return element_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict, inputs: dict, metrics: dict, *,
                    status: str) -> None:
    dt  = kw["dt"]
    nt  = kw["nt"]
    t_us = np.arange(nt) * dt * 1e6

    el_kw = compute_element_data(kw["p_raw"], inputs["element_kw_rows"], nt)
    el_py = compute_element_data(pkw_res["p_raw"], inputs["element_py_rows"], nt)

    vmax = float(max(np.abs(el_kw).max(), np.abs(el_py).max(), 1e-30))
    spacing = vmax * 2.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # k-wave stacked traces
    ax = axes[0]
    for i in range(NE):
        ax.plot(t_us, el_kw[i] + i * spacing, "b-", lw=0.7)
    ax.set_title("k-wave-python (13 elements)")
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("Element index")
    ax.set_yticks([i * spacing for i in range(NE)])
    ax.set_yticklabels([f"El {i+1}" for i in range(NE)], fontsize=7)

    # pykwavers stacked traces
    ax = axes[1]
    for i in range(NE):
        ax.plot(t_us, el_py[i] + i * spacing, "r-", lw=0.7)
    ax.set_title("pykwavers (13 elements)")
    ax.set_xlabel("Time [μs]")
    ax.set_yticks([i * spacing for i in range(NE)])
    ax.set_yticklabels([f"El {i+1}" for i in range(NE)], fontsize=7)

    # Per-element rms comparison
    ax = axes[2]
    rms_kw = np.sqrt(np.mean(el_kw**2, axis=1))
    rms_py = np.sqrt(np.mean(el_py**2, axis=1))
    elem_ids = np.arange(1, NE + 1)
    ax.bar(elem_ids - 0.2, rms_kw, 0.35, label="k-wave",     color="blue",  alpha=0.7)
    ax.bar(elem_ids + 0.2, rms_py, 0.35, label="pykwavers",  color="red",   alpha=0.7)
    ax.set_title("RMS per element")
    ax.set_xlabel("Element index")
    ax.set_ylabel("RMS pressure [Pa]")
    ax.legend(fontsize=8)
    ax.set_xticks(elem_ids)

    fig.suptitle(
        f"sd_directional_array_elements: k-wave vs pykwavers  [{status}]\n"
        f"{NX}×{NY}  dx={DX*1e3:.2f} mm  c={C0} m/s  t_end={T_END*1e6:.0f} μs  "
        f"{NE} elements  radius={SEMI_RADIUS}\n"
        f"pearson_r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for sd_directional_array_elements."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("sd_directional_array_elements: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}  dx={DX*1e3:.2f} mm")
    print(f"  Source : plane-wave row {SOURCE_ROW}  f={SOURCE_FREQ/1e6:.0f} MHz")
    print(f"  Sensor : {NE}-element semicircle  radius={SEMI_RADIUS} pts  arc=π")
    print(f"  PML    : inside  alpha=(2,0) [x active, y disabled]")
    print("=" * 70)

    print("\n[0/2] Building shared inputs (grid, source, sensor geometry)...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    n_active = inputs["n_active"]
    print(f"  Nt={nt}  dt={dt:.3e} s  n_active_sensors={n_active}")

    print("\n[1/2] k-wave-python (2-D PSTD, semicircular sensor)...")
    kw = run_kwave(inputs, no_cache=args.no_cache)
    kw_p = kw["p_raw"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa")

    print("\n[2/2] pykwavers (quasi-2D PSTD, semicircular sensor)...")
    pkw_res = run_pykwavers(inputs, no_cache=args.no_cache)
    py_p = pkw_res["p_raw"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa")

    # Compute per-element averaged data
    el_kw = compute_element_data(kw_p, inputs["element_kw_rows"], nt)
    el_py = compute_element_data(py_p, inputs["element_py_rows"], nt)

    # Parity on the (NE, Nt) element_data matrix
    metrics = compute_image_metrics(el_kw, el_py)

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics["psnr_db"]   >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"\n--- Parity (on 13-element data, shape {el_kw.shape}) ---")
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  {'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics['psnr_db']:.2f} dB  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    plot_comparison(kw, pkw_res, inputs, metrics, status=status)

    header = "\n".join([
        "sd_directional_array_elements parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  lossless",
        f"source: plane-wave row {SOURCE_ROW}  f={SOURCE_FREQ:.3e} Hz  filtered",
        f"sensor: {NE}-element semicircle  radius={SEMI_RADIUS} pts  n_active={n_active}",
        f"pml: inside  alpha=(2,0)",
        f"nt={nt}  dt={dt:.6e} s  t_end={T_END:.3e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ])
    report = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  (target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f} dB  (target >= {thr['psnr_db']} dB)",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(el_kw).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(el_py).max()):.6e}",
    ]
    save_text_report(METRICS_PATH, header, report)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
