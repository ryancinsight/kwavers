#!/usr/bin/env python3
"""
ivp_3D_simulation_compare.py
==============================
Parity comparison for the upstream ``ivp_3D_simulation`` example
(initial-value-problem with two ball sources in a 3-D heterogeneous medium).

Physical setup (matches k-wave-python ``examples/ivp_3D_simulation.py``):
  Grid:    64×64×64   dx = dy = dz = 0.1 mm
  Medium:  heterogeneous
             c[:32, :, :]  = 1800 m/s  (x < Nx/2)
             c[32:, :, :]  = 1500 m/s
             rho[:, 15:, :] = 1200 kg/m³  (y ≥ Ny/4)
             rho[:, :15, :] = 1000 kg/m³
  Source:  initial pressure p0 — two balls:
             ball-1: centre (38,32,32) 1-based, radius 5, magnitude 10 Pa
             ball-2: centre (20,20,20) 1-based, radius 3, magnitude 10 Pa
           Spatial smoothing applied (smooth_p0=True, default in k-wave).
  Sensor:  binary planar mask at z = Nz//2 = 32 (0-based index 31),
           recording all Nx×Ny = 4096 grid points as a time series.
  PML:     inside domain (default)

Comparison strategy
-------------------
We use a z-plane cross-section sensor rather than a full-grid sensor.
Full-grid time-series recording at 64³ × Nt ≈ 16.7 M float64 ≈ 130 MB —
the planar sensor reduces this to 4096 × Nt ≈ 2 M float64 ≈ 16 MB.

Sensor row ordering
-------------------
k-wave-python uses C-order (z changes fastest): row i → (i // Ny, i % Ny, 31).
pykwavers uses Fortran-order (x changes fastest): row j → (j % Nx, j // Nx, 31).
The standard C→Fortran permutation ``np.lexsort((active[:,0], active[:,1]))``
is applied to k-wave output before parity evaluation.

Outputs
-------
* ``output/ivp_3D_simulation_compare.png``
* ``output/ivp_3D_simulation_metrics.txt``

Usage
-----
  python examples/ivp_3D_simulation_compare.py
  python examples/ivp_3D_simulation_compare.py --no-cache
  python examples/ivp_3D_simulation_compare.py --allow-failure
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
# Physical constants  (must match ivp_3D_simulation.py exactly)
# ---------------------------------------------------------------------------
NX = NY = NZ = 64
DX = DY = DZ = 0.1e-3          # grid spacing [m]

C_BASE = 1500.0                 # [m/s]
C_FAST = 1800.0                 # x < Nx/2

RHO_BASE = 1000.0               # [kg/m³]
RHO_HIGH = 1200.0               # y ≥ Ny/4

BALL1_POS_1BASED = (38, 32, 32)
BALL1_RADIUS     = 5
BALL1_MAG        = 10.0         # [Pa]

BALL2_POS_1BASED = (20, 20, 20)
BALL2_RADIUS     = 3
BALL2_MAG        = 10.0         # [Pa]

SENSOR_Z_IDX = NZ // 2 - 1     # = 31  (0-based z-plane)
N_SENSORS    = NX * NY          # = 4096

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
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "ivp_3D_simulation_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "ivp_3D_simulation_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "ivp_3D_simulation_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "ivp_3D_simulation_pykwavers_cache.npz"

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
    """Build grid, medium arrays, p0 (smoothed), sensor mask, and permutation."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.utils.filters import smooth as kwave_smooth
    from kwave.utils.mapgen import make_ball

    # Sound speed and density maps
    c   = np.ones((NX, NY, NZ), dtype=np.float64) * C_BASE
    c[:32, :, :] = C_FAST

    rho = np.ones((NX, NY, NZ), dtype=np.float64) * RHO_BASE
    rho[:, 15:, :] = RHO_HIGH   # Ny/4=16 (1-based) → 15: (0-based)

    kgrid = kWaveGrid(Vector([NX, NY, NZ]), Vector([DX, DY, DZ]))
    kgrid.makeTime(c)            # uses max(c) = C_FAST for CFL dt

    # Initial pressure: two balls summed
    b1 = BALL1_MAG * np.asarray(make_ball(
        Vector([NX, NY, NZ]),
        Vector(list(BALL1_POS_1BASED)),
        BALL1_RADIUS,
    ), dtype=np.float64)
    b2 = BALL2_MAG * np.asarray(make_ball(
        Vector([NX, NY, NZ]),
        Vector(list(BALL2_POS_1BASED)),
        BALL2_RADIUS,
    ), dtype=np.float64)
    p0_raw = b1 + b2

    # Apply the same spatial smoothing that k-wave's smooth_p0=True applies
    p0_smooth = np.asarray(kwave_smooth(p0_raw, restore_max=True), dtype=np.float64)

    # Planar sensor at z = SENSOR_Z_IDX
    sensor_mask_3d = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_mask_3d[:, :, SENSOR_Z_IDX] = True

    # C-order → Fortran-order permutation for the z-plane
    # argwhere on (NX,NY,NZ) gives rows [x, y, z] in C-order (z-innermost)
    # For z=SENSOR_Z_IDX: C-order row i = (x=i//NY, y=i%NY, z=SENSOR_Z_IDX)
    # Fortran: x-innermost → perm = sort by (y primary, x secondary)
    active_c = np.argwhere(sensor_mask_3d)   # shape (4096, 3)
    perm = np.lexsort((active_c[:, 0], active_c[:, 1]))

    return {
        "c":            c,
        "rho":          rho,
        "p0_smooth":    p0_smooth,
        "kgrid":        kgrid,
        "sensor_mask":  sensor_mask_3d,
        "perm":         perm,
        "nt":           int(kgrid.Nt),
        "dt":           float(kgrid.dt),
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

    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    c   = inputs["c"]
    rho = inputs["rho"]
    kgrid = inputs["kgrid"]
    nt    = inputs["nt"]
    dt    = inputs["dt"]

    medium = kWaveMedium(sound_speed=c, density=rho)

    source   = kSource()
    source.p0 = inputs["p0_smooth"]

    sensor        = kSensor(mask=inputs["sensor_mask"], record=["p"])

    print(f"  [k-wave] Running 3-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        smooth_p0=False,     # smoothing already applied; pass raw p0_smooth
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
                f"Unexpected k-wave sensor shape {pressure.shape}; expected ({N_SENSORS}, {nt})"
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
        sound_speed=c[:, :, :].astype(np.float64),
        density=rho[:, :, :].astype(np.float64),
    )

    p0_3d  = inputs["p0_smooth"]
    source = pkw.Source.from_initial_pressure(p0_3d)

    sensor_mask_3d = inputs["sensor_mask"]
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running 3-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
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
                f"Unexpected pykwavers sensor shape {pressure.shape}; expected ({N_SENSORS}, {nt})"
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

    # Reshape to (NX, NY, Nt) for both using Fortran-order (x fastest)
    kw_img = kw_p.reshape(NX, NY, nt, order="F")
    py_img = py_p.reshape(NX, NY, nt, order="F")

    mid_t = nt // 2
    vmax  = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 0: snapshot at mid-timestep
    for ax, img, title in zip(axes[0], [kw_img[:, :, mid_t], py_img[:, :, mid_t],
                                          kw_img[:, :, mid_t] - py_img[:, :, mid_t]],
                               ["k-wave-python", "pykwavers", "difference"]):
        if "diff" in title:
            im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="seismic",
                           vmin=-vmax * 0.1, vmax=vmax * 0.1)
        else:
            im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="seismic",
                           vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title}  t={mid_t}")
        ax.set_xlabel("x [grid pts]")
        ax.set_ylabel("y [grid pts]")
        fig.colorbar(im, ax=ax)

    # Row 1: sensor trace at (NX//2, NY//2) ≡ Fortran row = NY//2 * NX + NX//2
    trace_idx = (NY // 2) * NX + NX // 2
    axes[1][0].plot(t_us, kw_p[trace_idx], "b-", lw=0.8, label="k-wave")
    axes[1][0].plot(t_us, py_p[trace_idx], "r--", lw=0.8, label="pykwavers")
    axes[1][0].set_title(f"Trace: sensor (Nx/2, Ny/2)")
    axes[1][0].set_xlabel("Time [μs]")
    axes[1][0].set_ylabel("Pressure [Pa]")
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(True, alpha=0.3)

    im4 = axes[1][1].imshow(kw_p, aspect="auto", origin="lower", cmap="seismic",
                             vmin=-vmax, vmax=vmax)
    axes[1][1].set_title("k-wave full sensor matrix")
    axes[1][1].set_xlabel("Time step")
    axes[1][1].set_ylabel("Sensor index (Fortran)")
    fig.colorbar(im4, ax=axes[1][1])

    im5 = axes[1][2].imshow(py_p, aspect="auto", origin="lower", cmap="seismic",
                             vmin=-vmax, vmax=vmax)
    axes[1][2].set_title("pykwavers full sensor matrix")
    axes[1][2].set_xlabel("Time step")
    axes[1][2].set_ylabel("Sensor index (Fortran)")
    fig.colorbar(im5, ax=axes[1][2])

    fig.suptitle(
        f"ivp_3D_simulation: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}×{NZ}  dx={DX*1e3:.2f} mm  "
        f"c=[{C_BASE},{C_FAST}] m/s  z-plane sensor (z={SENSOR_Z_IDX})\n"
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
        description="Compare pykwavers with k-wave-python for ivp_3D_simulation."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("ivp_3D_simulation: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}×{NZ}   dx={DX*1e3:.2f} mm")
    print(f"  Medium : c=[{C_BASE},{C_FAST}] m/s  rho=[{RHO_BASE},{RHO_HIGH}] kg/m³ (heterogeneous)")
    print(f"  Source : two balls IVP (p0)")
    print(f"  Sensor : z={SENSOR_Z_IDX} planar cross-section ({NX}×{NY} = {N_SENSORS} pts)")
    print("=" * 70)

    print("\n[0/2] Building shared inputs...")
    inputs = build_shared_inputs()
    nt, dt = inputs["nt"], inputs["dt"]
    print(f"  Nt={nt}  dt={dt:.3e} s")

    print("\n[1/2] k-wave-python (3-D PSTD, two-ball IVP)...")
    kw = run_kwave(inputs, no_cache=args.no_cache)
    kw_p = kw["pressure"]
    print(f"  shape={kw_p.shape}  peak={float(np.abs(kw_p).max()):.4e} Pa")

    print("\n[2/2] pykwavers (3-D PSTD, two-ball IVP)...")
    pkw_res = run_pykwavers(inputs, no_cache=args.no_cache)
    py_p = pkw_res["pressure"]
    print(f"  shape={py_p.shape}  peak={float(np.abs(py_p).max()):.4e} Pa")

    # Apply C→Fortran permutation to k-wave output
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

    kw_aligned_dict = {**kw, "pressure": kw_p_aligned}
    plot_comparison(kw_aligned_dict, pkw_res, metrics, status=status)

    header = "\n".join([
        "ivp_3D_simulation parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}x{NZ}  dx={DX:.6e} m",
        f"medium: c_fast={C_FAST} c_base={C_BASE} m/s  rho_high={RHO_HIGH} rho_base={RHO_BASE} kg/m3",
        f"source: two balls IVP  smooth_p0=True",
        f"sensor: z={SENSOR_Z_IDX} planar ({N_SENSORS} pts)",
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
