#!/usr/bin/env python3
"""
sd_directivity_modelling_3D_compare.py
========================================
Parity comparison for the upstream ``sd_directivity_modelling_3D`` example
(3-D sensor directivity — 11 point sources on a semicircular arc).

Physical setup (matches k-wave-python ``examples/sd_directivity_modelling_3D.py``):
  Grid:    64×64×64   dx = dy = dz = 100e-3/64 ≈ 1.5625 mm
  Medium:  homogeneous, c = 1500 m/s, lossless
  Source:  11 point sources equally spaced on a semicircular arc
           (radius = 20 grid pts, in the x-y plane at z = 0),
           each emitting a band-limited 0.25 MHz sinusoid.
           One separate simulation is run per source position.
  Sensor:  17×17 square face at x = Nx//2 = 32 (0-based),
           y = [24..40], z = [24..40] — 289 grid points.
           The 289 sensor time traces are summed to produce one directivity
           trace per source position.
  PML:     inside, pml_size = 10 grid points

Output comparison
-----------------
Both engines produce a (Nt, 11) ``single_element_data`` matrix.
Since the comparison metric is a sum over sensor points, the C-order /
Fortran-order sensor row convention does not affect the result:
``sensor_data.sum(axis=0)`` is commutative and ordering-independent.

Outputs
-------
* ``output/sd_directivity_modelling_3D_compare.png``
* ``output/sd_directivity_modelling_3D_metrics.txt``

Usage
-----
  python examples/sd_directivity_modelling_3D_compare.py
  python examples/sd_directivity_modelling_3D_compare.py --no-cache
  python examples/sd_directivity_modelling_3D_compare.py --allow-failure

Note
----
This example runs 11 independent 3-D PSTD simulations per engine.
Expect ~10–30 minutes wall time without cache.
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
# Physical constants  (must match sd_directivity_modelling_3D.py exactly)
# ---------------------------------------------------------------------------
NX = NY = NZ = 64
DX = DY = DZ = 100e-3 / NX     # ≈ 1.5625 mm

C0           = 1500.0           # [m/s]
SOURCE_FREQ  = 0.25e6           # [Hz]
SOURCE_MAG   = 1.0              # [Pa]
SOURCE_RADIUS = 20              # [grid points]
N_SOURCES    = 11               # number of arc source positions

PML_SIZE     = 10

# Sensor: 17×17 face at x=32 (0-based), y=[24..40], z=[24..40]
SZ           = 16               # half-span
SENSOR_X     = NX // 2         # = 32
SENSOR_Y_LO  = NY // 2 - SZ // 2   # = 24
SENSOR_Y_HI  = NY // 2 + SZ // 2 + 1  # = 41 (exclusive)
SENSOR_Z_LO  = NZ // 2 - SZ // 2   # = 24
SENSOR_Z_HI  = NZ // 2 + SZ // 2 + 1  # = 41 (exclusive)
N_SENSOR_PTS = 17 * 17          # = 289

F_MAX        = C0 / (2.0 * DX)  # ≈ 480 kHz — metadata for Source.from_mask

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
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "sd_directivity_modelling_3D_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "sd_directivity_modelling_3D_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "sd_directivity_3D_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "sd_directivity_3D_pykwavers_cache.npz"

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
            "element_data": np.asarray(d["element_data"], dtype=np.float64),
            "nt":           int(d["nt"]),
            "dt":           float(d["dt"]),
            "runtime_s":    float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_cache(path: os.PathLike, element_data: np.ndarray, nt: int,
                dt: float, runtime_s: float) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        element_data=np.asarray(element_data, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
def build_shared_inputs() -> dict:
    """Construct grid, medium, filtered signal, sensor mask, and 11 source positions."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.conversion import cart2grid
    from kwave.utils.filters import filter_time_series
    from kwave.utils.mapgen import make_cart_circle

    kgrid  = kWaveGrid(Vector([NX, NY, NZ]), Vector([DX, DY, DZ]))
    medium = kWaveMedium(sound_speed=C0)
    kgrid.makeTime(medium.sound_speed)

    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    t_array = np.asarray(kgrid.t_array).ravel()
    sig_raw = SOURCE_MAG * np.sin(2 * np.pi * SOURCE_FREQ * t_array)
    filtered = np.asarray(
        filter_time_series(kgrid, medium, sig_raw.reshape(1, -1)),
        dtype=np.float64,
    ).ravel()

    # Sensor mask: 17×17 face
    sensor_mask_3d = np.zeros((NX, NY, NZ), dtype=np.float64)
    sensor_mask_3d[SENSOR_X, SENSOR_Y_LO:SENSOR_Y_HI, SENSOR_Z_LO:SENSOR_Z_HI] = 1.0

    # 11 source positions: semicircular arc in x-y plane at z = 0
    kgrid_cart = kWaveGrid(Vector([NX, NY, NZ]), Vector([DX, DY, DZ]))
    circle_2d  = make_cart_circle(SOURCE_RADIUS * DX, N_SOURCES,
                                   center_pos=Vector([0, 0]), arc_angle=np.pi)
    circle_3d  = np.vstack([circle_2d, np.zeros((1, N_SOURCES))])

    circle_grid, _, _ = cart2grid(kgrid_cart, circle_3d, order="C")
    source_flat_indices = np.flatnonzero(circle_grid)   # C-order flat indices

    # Convert to (ix, iy, iz) 0-based tuples for both engines
    source_positions = []
    for flat_idx in source_flat_indices:
        ix, iy, iz = np.unravel_index(int(flat_idx), (NX, NY, NZ))
        source_positions.append((int(ix), int(iy), int(iz)))

    return {
        "kgrid":            kgrid,
        "medium":           medium,
        "filtered_sig":     filtered,
        "sensor_mask_3d":   sensor_mask_3d,
        "source_positions": source_positions,
        "source_flat_ids":  source_flat_indices,
        "nt":               nt,
        "dt":               dt,
    }


# ---------------------------------------------------------------------------
# k-wave-python run — 11 simulations
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
    n_pts    = N_SENSOR_PTS
    sensor_mask_3d = inputs["sensor_mask_3d"]

    element_data = np.zeros((nt, N_SOURCES), dtype=np.float64)

    t0 = time.perf_counter()
    for i, flat_idx in enumerate(inputs["source_flat_ids"]):
        p_mask = np.zeros((NX, NY, NZ), dtype=np.float64)
        p_mask.flat[flat_idx] = 1.0

        source        = kSource()
        source.p_mask = p_mask
        source.p      = inputs["filtered_sig"].reshape(1, -1)

        sensor = kSensor(mask=sensor_mask_3d.astype(bool), record=["p"])

        result = kspaceFirstOrder(
            kgrid, medium, source, sensor,
            pml_inside=True,
            pml_size=PML_SIZE,
            backend="python",
            device="cpu",
            quiet=True,
        )
        p = np.asarray(result["p"], dtype=np.float64)
        if p.shape[0] != n_pts:
            if p.shape[1] == n_pts:
                p = p.T
        element_data[:, i] = p.sum(axis=0)

        elapsed = time.perf_counter() - t0
        print(f"    source {i+1:2d}/{N_SOURCES}  {elapsed:.0f}s elapsed")

    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done: {runtime_s:.1f} s total")

    _save_cache(_KWAVE_CACHE, element_data, nt, dt, runtime_s)
    return {"element_data": element_data, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# pykwavers run — 11 simulations
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    nt = inputs["nt"]
    dt = inputs["dt"]

    grid   = pkw.Grid(NX, NY, NZ, DX, DY, DZ)
    medium = pkw.Medium(sound_speed=C0)

    sensor_mask_3d = inputs["sensor_mask_3d"].astype(bool)
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)

    element_data = np.zeros((nt, N_SOURCES), dtype=np.float64)

    t0 = time.perf_counter()
    for i, (ix, iy, iz) in enumerate(inputs["source_positions"]):
        src_mask_3d = np.zeros((NX, NY, NZ), dtype=np.float64)
        src_mask_3d[ix, iy, iz] = 1.0

        filtered = inputs["filtered_sig"]  # 1D — single source
        source   = pkw.Source.from_mask(src_mask_3d, filtered, F_MAX)

        sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
        sim.set_pml_inside(True)
        sim.set_pml_size(PML_SIZE)

        result = sim.run(time_steps=nt, dt=dt)
        p = np.asarray(result.sensor_data, dtype=np.float64)
        if p.shape[0] != N_SENSOR_PTS:
            if p.shape[1] == N_SENSOR_PTS:
                p = p.T

        # Sum over sensor points (ordering-independent)
        element_data[:, i] = p.sum(axis=0)

        elapsed = time.perf_counter() - t0
        print(f"    source {i+1:2d}/{N_SOURCES}  {elapsed:.0f}s elapsed")

    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done: {runtime_s:.1f} s total")

    _save_cache(_PKWAV_CACHE, element_data, nt, dt, runtime_s)
    return {"element_data": element_data, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict, inputs: dict, metrics: dict, *,
                    status: str) -> None:
    el_kw = kw["element_data"]    # (Nt, 11)
    el_py = pkw_res["element_data"]
    nt    = kw["nt"]
    dt    = kw["dt"]
    t_us  = np.arange(nt) * dt * 1e6

    # Directivity peak values
    peak_kw = np.max(np.abs(el_kw), axis=0)
    peak_py = np.max(np.abs(el_py), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Time series overlay for central source (index 5 ~ broadside)
    mid = N_SOURCES // 2
    axes[0].plot(t_us, el_kw[:, mid], "b-",  lw=0.8, label="k-wave")
    axes[0].plot(t_us, el_py[:, mid], "r--", lw=0.8, label="pykwavers")
    axes[0].set_title(f"Broadside trace (source {mid+1}/{N_SOURCES})")
    axes[0].set_xlabel("Time [μs]")
    axes[0].set_ylabel("Summed pressure [Pa]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Directivity comparison
    src_ids = np.arange(1, N_SOURCES + 1)
    axes[1].bar(src_ids - 0.2, peak_kw, 0.35, label="k-wave",    color="blue",  alpha=0.7)
    axes[1].bar(src_ids + 0.2, peak_py, 0.35, label="pykwavers", color="red",   alpha=0.7)
    axes[1].set_title("Directivity: peak |p| per source angle")
    axes[1].set_xlabel("Source position index")
    axes[1].set_ylabel("Peak summed pressure [Pa]")
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(src_ids)

    # Full element_data imshow comparison
    vmax = float(max(np.abs(el_kw).max(), np.abs(el_py).max(), 1e-30))
    diff = el_py - el_kw
    im = axes[2].imshow(diff.T, aspect="auto", origin="lower", cmap="seismic",
                        vmin=-vmax * 0.1, vmax=vmax * 0.1)
    axes[2].set_title("Residual: pykwavers − k-wave  (11 × Nt)")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Source position index")
    fig.colorbar(im, ax=axes[2])

    fig.suptitle(
        f"sd_directivity_modelling_3D: k-wave vs pykwavers  [{status}]\n"
        f"Grid {NX}³  dx={DX*1e3:.2f} mm  c={C0} m/s  f={SOURCE_FREQ/1e3:.0f} kHz  "
        f"{N_SOURCES} sources  17×17 sensor face\n"
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
        description="Compare pykwavers with k-wave-python for sd_directivity_modelling_3D."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("sd_directivity_modelling_3D: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}³  dx={DX*1e3:.2f} mm")
    print(f"  Source : {N_SOURCES} pts on semicircle  r={SOURCE_RADIUS} pts  f={SOURCE_FREQ/1e3:.0f} kHz")
    print(f"  Sensor : 17×17 face at x={SENSOR_X}  ({N_SENSOR_PTS} pts)  pml_size={PML_SIZE}")
    print(f"  NOTE   : Runs {N_SOURCES} × 3D PSTD sims per engine — may take ~10-30 min without cache")
    print("=" * 70)

    print("\n[0/2] Building shared inputs...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    print(f"  Nt={nt}  dt={dt:.3e} s")
    print(f"  Source positions (ix,iy,iz): {inputs['source_positions']}")

    print(f"\n[1/2] k-wave-python ({N_SOURCES} × 3-D PSTD)...")
    kw = run_kwave(inputs, no_cache=args.no_cache)
    el_kw = kw["element_data"]
    print(f"  shape={el_kw.shape}  peak={float(np.abs(el_kw).max()):.4e} Pa")

    print(f"\n[2/2] pykwavers ({N_SOURCES} × 3-D PSTD)...")
    pkw_res = run_pykwavers(inputs, no_cache=args.no_cache)
    el_py = pkw_res["element_data"]
    print(f"  shape={el_py.shape}  peak={float(np.abs(el_py).max()):.4e} Pa")

    # Parity on (Nt, 11) element_data matrix
    metrics = compute_image_metrics(el_kw, el_py)

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
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    plot_comparison(kw, pkw_res, inputs, metrics, status=status)

    header = "\n".join([
        "sd_directivity_modelling_3D parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}x{NZ}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  lossless",
        f"source: {N_SOURCES} pts on semicircle r={SOURCE_RADIUS} pts  f={SOURCE_FREQ:.3e} Hz",
        f"sensor: 17x17 face at x={SENSOR_X}  ({N_SENSOR_PTS} pts)  pml_size={PML_SIZE}",
        f"nt={nt}  dt={dt:.6e} s",
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
        f"peak_ratio        = {metrics['peak_ratio']:.6f}",
    ]
    save_text_report(METRICS_PATH, header, report)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
