#!/usr/bin/env python3
"""
tvsp_doppler_effect_compare.py
================================
Parity comparison for the upstream ``tvsp_doppler_effect`` example
(moving pressure source demonstrating the Doppler effect).

Physical setup (matches k-wave-python ``examples/tvsp_doppler_effect.py``):
  Grid:    64×128   dx = dy = 20e-3/128 ≈ 0.15625 mm
  Medium:  homogeneous with absorption
             c           = 1500 m/s
             alpha_coeff = 0.75 dB/(MHz^y cm)
             alpha_power = 1.5
  Time:    Nt = 4500,  dt = 20 ns  (manually set, not via makeTime)
  Source:  row of 88 points at x = Nx - 1 - pml_size - source_x_pos = 38 (0-based),
           y = pml_size : Ny - pml_size = 20:108 (0-based).
           A 0.75 MHz filtered sinusoid is distributed between adjacent grid
           points at each time step via linear interpolation to simulate a
           source moving at 150 m/s in the y-direction.
           Pre-computed source_p matrix has shape (88, Nt).
  Sensor:  single point at (33, 63) 0-based — records the frequency-shifted
           pressure as the source passes.
  PML:     inside, pml_size = 20 grid points

Moving-source construction
--------------------------
The same source_p matrix is computed once from k-wave's time array and used
by both engines.  All 88 source positions are in row 38 (fixed x), so C-order
and Fortran-order enumeration of the source mask both yield the same sequence
(y ascending within fixed x = same ordering in both conventions).

Sensor ordering
---------------
Single-point sensor at (33, 63) — no permutation ambiguity.

Outputs
-------
* ``output/tvsp_doppler_effect_compare.png``
* ``output/tvsp_doppler_effect_metrics.txt``

Usage
-----
  python examples/tvsp_doppler_effect_compare.py
  python examples/tvsp_doppler_effect_compare.py --no-cache
  python examples/tvsp_doppler_effect_compare.py --allow-failure
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
    compute_trace_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical constants  (must match tvsp_doppler_effect.py exactly)
# ---------------------------------------------------------------------------
NX       = 64
NY       = NX * 2               # = 128
DY       = 20e-3 / NY           # ≈ 0.15625 mm
DX       = DY
PML_SIZE = 20

C0          = 1500.0            # [m/s]
ALPHA_COEFF = 0.75              # [dB/(MHz^y cm)]
ALPHA_POWER = 1.5

SOURCE_VEL    = 150.0           # [m/s]
SOURCE_FREQ   = 0.75e6          # [Hz]
SOURCE_MAG    = 3.0             # [Pa]
SOURCE_X_POS  = 5               # [grid points] from right PML edge

# 0-based source row and y-range
SOURCE_ROW = NX - 1 - PML_SIZE - SOURCE_X_POS  # = 38
SOURCE_Y_LO = PML_SIZE                           # = 20 (inclusive)
SOURCE_Y_HI = NY - PML_SIZE                      # = 108 (exclusive)
N_SOURCE     = SOURCE_Y_HI - SOURCE_Y_LO         # = 88

# Fixed time parameters
NT = 4500
DT = 20e-9                      # [s]

# Sensor position (0-based)
SOURCE_SENSOR_X_DIST = 5
SENSOR_X = NX - 1 - PML_SIZE - SOURCE_X_POS - SOURCE_SENSOR_X_DIST  # = 33
SENSOR_Y = NY // 2 - 1                                                # = 63

F_MAX = C0 / (2.0 * DX)        # metadata for Source.from_mask

# ---------------------------------------------------------------------------
# Parity thresholds
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.90,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.25,
    "psnr_db":       15.0,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "tvsp_doppler_effect_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_doppler_effect_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_doppler_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_doppler_pykwavers_cache.npz"

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
# Shared source construction
# ---------------------------------------------------------------------------
def _build_source_p() -> tuple[np.ndarray, np.ndarray]:
    """Construct the moving-source pressure matrix and sensor mask.

    Returns
    -------
    source_p   : (N_SOURCE, NT) float64 — pre-computed moving-source signal
    p_mask_2d  : (NX, NY) float64 — source mask with row SOURCE_ROW active

    The 0.75 MHz sinusoidal pressure is distributed between adjacent grid
    points via linear interpolation at each time step, simulating a source
    travelling at SOURCE_VEL = 150 m/s in the y-direction.

    Proof of ordering invariance
    ----------------------------
    All N_SOURCE = 88 active source points lie in row x=SOURCE_ROW.
    In a (NX, NY) mask:
      C-order flat index   = x * NY + y  →  SOURCE_ROW * NY + (SOURCE_Y_LO..SOURCE_Y_HI-1)
      Fortran-order position among active points: sorted by x, then y within same x
        = SOURCE_ROW * (SOURCE_Y_HI-SOURCE_Y_LO) + (0..N_SOURCE-1)
    Both orderings agree within the single-row mask (y increases in both).
    Hence source_p rows 0..N_SOURCE-1 correspond to y=20..107 in both engines.
    """
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.filters import filter_time_series

    kgrid  = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    kgrid.setTime(NT, DT)
    medium = kWaveMedium(
        sound_speed=C0, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER
    )

    t_array = np.asarray(kgrid.t_array).ravel()
    source_pressure_raw = SOURCE_MAG * np.sin(2 * np.pi * SOURCE_FREQ * t_array)
    source_pressure = np.asarray(
        filter_time_series(kgrid, medium, source_pressure_raw.reshape(1, -1)),
        dtype=np.float64,
    ).ravel()

    p_mask_2d = np.zeros((NX, NY), dtype=np.float64)
    p_mask_2d[SOURCE_ROW, SOURCE_Y_LO:SOURCE_Y_HI] = 1.0

    source_p = np.zeros((N_SOURCE, NT), dtype=np.float64)

    sensor_index = 0
    t_index      = 0
    while t_index < NT and sensor_index < N_SOURCE - 2:
        if t_array[t_index] > ((sensor_index + 1) * DY / SOURCE_VEL):
            sensor_index += 1
        exact_pos    = SOURCE_VEL * t_array[t_index]
        discrete_pos = (sensor_index + 1) * DY
        pos_ratio    = (discrete_pos - exact_pos) / DY

        source_p[sensor_index,     t_index] = pos_ratio       * source_pressure[t_index]
        source_p[sensor_index + 1, t_index] = (1.0 - pos_ratio) * source_pressure[t_index]
        t_index += 1

    return source_p, p_mask_2d


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(source_p: np.ndarray, p_mask_2d: np.ndarray,
              *, no_cache: bool = False) -> dict:
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

    kgrid  = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    kgrid.setTime(NT, DT)
    medium = kWaveMedium(
        sound_speed=C0, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER
    )

    source        = kSource()
    source.p_mask = p_mask_2d
    source.p      = source_p   # (N_SOURCE, NT)

    sensor_mask_2d = np.zeros((NX, NY), dtype=float)
    sensor_mask_2d[SENSOR_X, SENSOR_Y] = 1.0
    sensor = kSensor(mask=sensor_mask_2d, record=["p"])

    print(f"  [k-wave] Running 2-D PSTD Doppler  (Nt={NT}, dt={DT:.1e} s)...")
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

    pressure = np.asarray(result["p"], dtype=np.float64).ravel()
    _save_cache(_KWAVE_CACHE, pressure, runtime_s)
    return {"pressure": pressure, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(source_p: np.ndarray, p_mask_2d: np.ndarray,
                  *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    grid   = pkw.Grid(NX, NY, 1, DX, DY, DX)
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=1000.0,
        absorption=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    # Expand source mask to 3D; source_p rows match active points in both
    # C-order and Fortran-order (all points in one row of fixed x).
    src_mask_3d = p_mask_2d[:, :, None]
    source = pkw.Source.from_mask(src_mask_3d, source_p, F_MAX)

    sensor_mask_3d = np.zeros((NX, NY, 1), dtype=bool)
    sensor_mask_3d[SENSOR_X, SENSOR_Y, 0] = True
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running quasi-2D PSTD Doppler  (Nt={NT}, dt={DT:.1e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64).ravel()
    _save_cache(_PKWAV_CACHE, pressure, runtime_s)
    return {"pressure": pressure, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict, metrics: dict, *,
                    status: str) -> None:
    kw_p  = kw["pressure"]
    py_p  = pkw_res["pressure"]
    t_us  = np.arange(NT) * DT * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_us, kw_p, "b-",  lw=0.8, label="k-wave-python")
    axes[0].plot(t_us, py_p, "r--", lw=0.8, label="pykwavers")
    axes[0].set_title(f"Sensor ({SENSOR_X},{SENSOR_Y}) — Doppler-shifted trace")
    axes[0].set_xlabel("Time [μs]")
    axes[0].set_ylabel("Pressure [Pa]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_us, py_p - kw_p, "k-", lw=0.8)
    axes[1].set_title("Residual: pykwavers − k-wave")
    axes[1].set_xlabel("Time [μs]")
    axes[1].set_ylabel("ΔPressure [Pa]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"tvsp_doppler_effect: k-wave-python vs pykwavers  [{status}]\n"
        f"{NX}×{NY}  dx={DX*1e3:.3f} mm  source_vel={SOURCE_VEL} m/s  "
        f"f={SOURCE_FREQ/1e6:.2f} MHz  Nt={NT}\n"
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
        description="Compare pykwavers with k-wave-python for tvsp_doppler_effect."
    )
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("tvsp_doppler_effect: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}  dx={DX*1e3:.3f} mm")
    print(f"  Source : moving at {SOURCE_VEL} m/s  f={SOURCE_FREQ/1e6:.2f} MHz  "
          f"row={SOURCE_ROW} y={SOURCE_Y_LO}..{SOURCE_Y_HI-1}  ({N_SOURCE} pts)")
    print(f"  Sensor : ({SENSOR_X},{SENSOR_Y})  Nt={NT}  dt={DT:.1e} s")
    print("=" * 70)

    print("\n[0/2] Building shared moving-source matrix  (loop over 4500 steps)...")
    t_build = time.perf_counter()
    source_p, p_mask_2d = _build_source_p()
    print(f"  Built in {time.perf_counter()-t_build:.1f} s  source_p.shape={source_p.shape}")

    print("\n[1/2] k-wave-python (2-D PSTD, Doppler moving source)...")
    kw = run_kwave(source_p, p_mask_2d, no_cache=args.no_cache)
    kw_p = kw["pressure"]
    print(f"  len={len(kw_p)}  peak={float(np.abs(kw_p).max()):.4e} Pa")

    print("\n[2/2] pykwavers (quasi-2D PSTD, Doppler moving source)...")
    pkw_res = run_pykwavers(source_p, p_mask_2d, no_cache=args.no_cache)
    py_p = pkw_res["pressure"]
    print(f"  len={len(py_p)}  peak={float(np.abs(py_p).max()):.4e} Pa")

    metrics = compute_trace_metrics(kw_p, py_p)

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics["psnr_db"]    >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"\n--- Parity ---")
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  {'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics['psnr_db']:.2f} dB  {'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    plot_comparison(kw, pkw_res, metrics, status=status)

    header = "\n".join([
        "tvsp_doppler_effect parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c={C0} m/s  alpha={ALPHA_COEFF}  power={ALPHA_POWER}",
        f"source: moving at {SOURCE_VEL} m/s  f={SOURCE_FREQ:.3e} Hz  row={SOURCE_ROW}",
        f"sensor: ({SENSOR_X},{SENSOR_Y})",
        f"nt={NT}  dt={DT:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ])
    report = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  (target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f} dB  (target >= {thr['psnr_db']} dB)",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw_p).max()):.6e}",
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
