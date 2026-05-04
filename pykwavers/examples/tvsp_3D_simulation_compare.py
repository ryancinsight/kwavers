#!/usr/bin/env python3
"""
tvsp_3D_simulation_compare.py
==============================
Parity comparison for the upstream ``tvsp_3D_simulation`` example
(3-D time-varying sinusoidal source in a heterogeneous medium).

Physical setup (matches k-wave-python ``examples/tvsp_3D_simulation.py``):
  Grid:    64×64×64   dx = dy = dz = 0.1 mm
  Medium:  heterogeneous
             c[:Nx/2, :, :]     = 1800 m/s  (x < 32)
             c[Nx/2:, :, :]     = 1500 m/s
             rho[:, Ny/4-1:, :] = 1200 kg/m³  (y ≥ 15, 0-based)
             rho[:, :Ny/4-1, :] = 1000 kg/m³
  Source:  square 11×11 patch at x=15 (0-based), y=26..36, z=26..36,
           emitting a band-limited 2 MHz sinusoid (shared filtered signal).
  Sensor:  binary planar mask at z = Nz//2 - 1 = 31 (0-based),
           recording all Nx×Ny = 4096 grid points as a time series.
  PML:     inside domain (default)

Sensor row ordering
-------------------
Same C→Fortran permutation as ivp_3D_simulation_compare:
``np.lexsort((active[:,0], active[:,1]))`` applied to k-wave output.

Outputs
-------
* ``output/tvsp_3D_simulation_compare.png``
* ``output/tvsp_3D_simulation_metrics.txt``

Usage
-----
  python examples/tvsp_3D_simulation_compare.py
  python examples/tvsp_3D_simulation_compare.py --no-cache
  python examples/tvsp_3D_simulation_compare.py --allow-failure
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
# Physical constants  (must match tvsp_3D_simulation.py exactly)
# ---------------------------------------------------------------------------
NX = NY = NZ = 64
DX = DY = DZ = 0.1e-3          # grid spacing [m]

C_BASE = 1500.0
C_FAST = 1800.0                 # x < Nx/2

RHO_BASE = 1000.0
RHO_HIGH = 1200.0               # y ≥ Ny/4 - 1 (0-based)

SOURCE_FREQ = 2e6               # [Hz]
SOURCE_MAG  = 1.0               # [Pa]
SOURCE_RADIUS = 5               # [grid points] → 11×11 patch

# Source patch (0-based):
#   cx = Nx//4 - 1 = 15
#   y_lo = Ny//2 - source_radius - 1 = 26
#   y_hi = Ny//2 + source_radius     = 37  (exclusive)
#   z_lo = Nz//2 - source_radius - 1 = 26
#   z_hi = Nz//2 + source_radius     = 37  (exclusive)
SOURCE_CX  = NX // 4 - 1       # = 15
SOURCE_Y_LO = NY // 2 - SOURCE_RADIUS - 1  # = 26
SOURCE_Y_HI = NY // 2 + SOURCE_RADIUS      # = 37
SOURCE_Z_LO = NZ // 2 - SOURCE_RADIUS - 1  # = 26
SOURCE_Z_HI = NZ // 2 + SOURCE_RADIUS      # = 37

SENSOR_Z_IDX = NZ // 2 - 1     # = 31  (0-based z-plane)
N_SENSORS    = NX * NY          # = 4096

F_MAX = C_BASE / (2.0 * DX)    # ≈ 7.5 MHz — metadata for Source.from_mask

# ---------------------------------------------------------------------------
# Parity thresholds
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.95,
    "rms_ratio_min": 0.75,
    "rms_ratio_max": 1.30,
    "psnr_db":       18.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "tvsp_3D_simulation_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_3D_simulation_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_3D_simulation_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_3D_simulation_pykwavers_cache.npz"

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
            "nt":        int(d["nt"]),
            "dt":        float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_cache(path: os.PathLike, pressure: np.ndarray, nt: int,
                dt: float, runtime_s: float) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        pressure=np.asarray(pressure, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
def build_shared_inputs() -> dict:
    """Build grid, medium, filtered source signal, sensor mask, and permutation."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.filters import filter_time_series

    c   = np.ones((NX, NY, NZ), dtype=np.float64) * C_BASE
    c[:NX // 2, :, :] = C_FAST

    rho = np.ones((NX, NY, NZ), dtype=np.float64) * RHO_BASE
    rho[:, NY // 4 - 1:, :] = RHO_HIGH   # 0-based index 15

    kgrid  = kWaveGrid(Vector([NX, NY, NZ]), Vector([DX, DY, DZ]))
    medium = kWaveMedium(sound_speed=c, density=rho)
    kgrid.makeTime(c)

    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    t_array = np.asarray(kgrid.t_array).ravel()
    sig_raw = SOURCE_MAG * np.sin(2 * np.pi * SOURCE_FREQ * t_array)

    # filter_time_series expects (num_signals, Nt) — use (1, Nt) then ravel
    filtered = np.asarray(
        filter_time_series(kgrid, medium, sig_raw.reshape(1, -1)),
        dtype=np.float64,
    ).ravel()   # shape (Nt,)

    # Source mask: 11×11 patch at (cx, y_lo:y_hi, z_lo:z_hi)
    src_mask_3d = np.zeros((NX, NY, NZ), dtype=np.float64)
    src_mask_3d[SOURCE_CX, SOURCE_Y_LO:SOURCE_Y_HI, SOURCE_Z_LO:SOURCE_Z_HI] = 1.0
    n_source = int(src_mask_3d.sum())   # = 121

    # Planar sensor at z = SENSOR_Z_IDX
    sensor_mask_3d = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask_3d[:, :, SENSOR_Z_IDX] = True

    active_c = np.argwhere(sensor_mask_3d)   # (4096, 3)
    perm     = np.lexsort((active_c[:, 0], active_c[:, 1]))

    return {
        "c":            c,
        "rho":          rho,
        "kgrid":        kgrid,
        "medium":       medium,
        "filtered_sig": filtered,
        "src_mask_3d":  src_mask_3d,
        "n_source":     n_source,
        "sensor_mask":  sensor_mask_3d,
        "perm":         perm,
        "nt":           nt,
        "dt":           dt,
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

    kgrid  = inputs["kgrid"]
    medium = inputs["medium"]
    nt     = inputs["nt"]
    dt     = inputs["dt"]

    source        = kSource()
    source.p_mask = inputs["src_mask_3d"].astype(float)
    # Broadcast the 1D filtered signal to all 121 source points
    source.p      = inputs["filtered_sig"].reshape(1, -1)   # (1, Nt)

    sensor = kSensor(mask=inputs["sensor_mask"], record=["p"])

    print(f"  [k-wave] Running 3-D PSTD  (Nt={nt}, dt={dt:.3e} s, n_src={inputs['n_source']})...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        pml_inside=True,
        backend="python",
        device="cpu",
        quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result["p"], dtype=np.float64)
    if pressure.shape[0] != N_SENSORS:
        if pressure.shape[1] == N_SENSORS:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected k-wave shape {pressure.shape}; expected ({N_SENSORS}, {nt})"
            )

    _save_cache(_KWAVE_CACHE, pressure, nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    c   = inputs["c"]
    rho = inputs["rho"]
    nt  = inputs["nt"]
    dt  = inputs["dt"]

    grid   = pkw.Grid(NX, NY, NZ, DX, DY, DZ)
    medium = pkw.Medium(
        sound_speed=c.astype(np.float64),
        density=rho.astype(np.float64),
    )

    src_mask_3d = inputs["src_mask_3d"]
    filtered    = inputs["filtered_sig"]  # (Nt,) 1D — broadcast to all 121 sources
    source = pkw.Source.from_mask(src_mask_3d, filtered, F_MAX)

    sensor = pkw.Sensor.from_mask(inputs["sensor_mask"])

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running 3-D PSTD  (Nt={nt}, dt={dt:.3e} s, n_src={inputs['n_source']})...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64)
    if pressure.shape[0] != N_SENSORS:
        if pressure.shape[1] == N_SENSORS:
            pressure = pressure.T
        else:
            raise AssertionError(
                f"Unexpected pykwavers shape {pressure.shape}; expected ({N_SENSORS}, {nt})"
            )

    _save_cache(_PKWAV_CACHE, pressure, nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw_aligned: dict, pkw_res: dict, metrics: dict, *,
                    status: str) -> None:
    kw_p = kw_aligned["pressure"]
    py_p = pkw_res["pressure"]
    nt   = kw_aligned["nt"]
    dt   = kw_aligned["dt"]
    t_us = np.arange(nt) * dt * 1e6

    vmax  = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))

    # Reshape to (NX, NY, Nt) Fortran-order for spatial snapshots
    kw_img = kw_p.reshape(NX, NY, nt, order="F")
    py_img = py_p.reshape(NX, NY, nt, order="F")
    mid_t  = nt // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, img, title in zip(axes[0],
                               [kw_img[:, :, mid_t], py_img[:, :, mid_t],
                                kw_img[:, :, mid_t] - py_img[:, :, mid_t]],
                               ["k-wave-python", "pykwavers", "difference"]):
        scale = vmax * 0.1 if "diff" in title else vmax
        im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="seismic",
                       vmin=-scale, vmax=scale)
        ax.set_title(f"{title}  t_step={mid_t}")
        ax.set_xlabel("x [grid pts]")
        ax.set_ylabel("y [grid pts]")
        fig.colorbar(im, ax=ax)

    # Centre trace
    trace_idx = (NY // 2) * NX + NX // 2
    axes[1][0].plot(t_us, kw_p[trace_idx], "b-", lw=0.8, label="k-wave")
    axes[1][0].plot(t_us, py_p[trace_idx], "r--", lw=0.8, label="pykwavers")
    axes[1][0].set_title(f"Trace: centre (Nx/2, Ny/2, z={SENSOR_Z_IDX})")
    axes[1][0].set_xlabel("Time [μs]")
    axes[1][0].set_ylabel("Pressure [Pa]")
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(True, alpha=0.3)

    im4 = axes[1][1].imshow(kw_p, aspect="auto", origin="lower", cmap="seismic",
                             vmin=-vmax, vmax=vmax)
    axes[1][1].set_title("k-wave sensor matrix")
    axes[1][1].set_xlabel("Time step")
    axes[1][1].set_ylabel("Sensor index")
    fig.colorbar(im4, ax=axes[1][1])

    im5 = axes[1][2].imshow(py_p, aspect="auto", origin="lower", cmap="seismic",
                             vmin=-vmax, vmax=vmax)
    axes[1][2].set_title("pykwavers sensor matrix")
    axes[1][2].set_xlabel("Time step")
    axes[1][2].set_ylabel("Sensor index")
    fig.colorbar(im5, ax=axes[1][2])

    fig.suptitle(
        f"tvsp_3D_simulation: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}³  dx={DX*1e3:.2f} mm  f={SOURCE_FREQ/1e6:.0f} MHz  "
        f"z-plane sensor (z={SENSOR_Z_IDX})\n"
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
        description="Compare pykwavers with k-wave-python for tvsp_3D_simulation."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("tvsp_3D_simulation: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}³   dx={DX*1e3:.2f} mm")
    print(f"  Source : {SOURCE_RADIUS*2+1}×{SOURCE_RADIUS*2+1} patch at x={SOURCE_CX}  f={SOURCE_FREQ/1e6:.0f} MHz")
    print(f"  Sensor : z={SENSOR_Z_IDX} plane ({N_SENSORS} pts)")
    print("=" * 70)

    print("\n[0/2] Building shared inputs...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    print(f"  Nt={nt}  dt={dt:.3e} s  n_source={inputs['n_source']}")

    print("\n[1/2] k-wave-python (3-D PSTD, square-patch TVSP)...")
    kw = run_kwave(inputs, no_cache=args.no_cache)
    kw_p = kw["pressure"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa")

    print("\n[2/2] pykwavers (3-D PSTD, square-patch TVSP)...")
    pkw_res = run_pykwavers(inputs, no_cache=args.no_cache)
    py_p = pkw_res["pressure"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa")

    perm = inputs["perm"]
    kw_p_aligned = kw_p[perm]

    metrics = compute_image_metrics(kw_p_aligned, py_p)

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

    plot_comparison({**kw, "pressure": kw_p_aligned}, pkw_res, metrics, status=status)

    header = "\n".join([
        "tvsp_3D_simulation parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}x{NZ}  dx={DX:.6e} m",
        f"medium: c=[{C_BASE},{C_FAST}] m/s  rho=[{RHO_BASE},{RHO_HIGH}] kg/m3 (heterogeneous)",
        f"source: 11x11 patch at x={SOURCE_CX}  f={SOURCE_FREQ:.3e} Hz  mag={SOURCE_MAG} Pa",
        f"sensor: z={SENSOR_Z_IDX} plane ({N_SENSORS} pts)",
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
        f"peak_kwave_Pa     = {float(np.abs(kw_p_aligned).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(py_p).max()):.6e}",
    ]
    save_text_report(METRICS_PATH, header, report)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
