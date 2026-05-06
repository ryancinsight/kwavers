#!/usr/bin/env python3
"""
na_modelling_absorption_compare.py
===================================
Side-by-side parity comparison for the upstream k-Wave example
``example_na_modelling_absorption.m`` (case 2 of 3): a 1-D
power-law-absorbing medium with a delta initial-pressure pulse and two
sensors at fixed offsets, used in the original example to characterise
the attenuation α(f) = α₀·f^y dispersion encapsulated by k-Wave's
fractional-Laplacian absorption operator.

Physical setup (matches MATLAB example_na_modelling_absorption.m, case 2)
------------------------------------------------------------------------
Grid    : 1024 grid points × 1 × 1, ``dx = 12.8 mm / 1024 ≈ 12.5 µm``
Medium  : c = 1500 m/s, ρ = 1000 kg/m³,
          α = 0.25 dB/(MHz^y cm), y = 1.5 (middle test case)
Source  : initial pressure delta at x = Nx/4
Sensors : two binary sensor points at offsets
          ``+round(0.5e-3 / dx)`` and ``+round(1.5e-3 / dx)`` from source
Time    : default ``kgrid.makeTime(c, [], 4 µs)``

Comparison strategy
-------------------
Both engines run with identical absorption parameters and identical
delta-IVP source. The 2-sensor pressure traces are compared with
image-level Pearson r, RMS ratio, PSNR. A 4-panel figure shows:
  1. Sensor 1 trace (kw vs pyk overlay)
  2. Sensor 2 trace (kw vs pyk overlay)
  3. Both engines' sensor matrix (2 × Nt)
  4. Difference

Outputs
-------
* ``output/na_modelling_absorption_compare.png``
* ``output/na_modelling_absorption_metrics.txt``

Why this is in scope
--------------------
The ``ivp_loading_external_image_compare.py`` debugging session
identified that pykwavers' PSTD has a propagation-distance-dependent
dispersion drift vs k-wave-python's. This 1-D absorption test is a
clean isolation — single delta pulse, two sensors at known distances —
and reveals whether the drift surfaces in 1-D as well, which would
help narrow the kspace-correction audit (filed as a separate task).

Status: **NEAR-PASS** — peak amplitude and RMS now match k-wave-python
exactly; residual Pearson is the well-characterised 1-sample timing
offset between the two engines on a sub-cycle pulse.

   pearson_r  = 0.917   (target ≥ 0.97)  — fails by Δ=0.05 due to 1-dt phase
   rms_ratio  = 1.0000  (target [0.85, 1.20])  — PASS
   psnr_db    = 35.3 dB (target ≥ 18.0 dB)  — PASS
   peak_kw    = 0.3836 Pa
   peak_pyk   = 0.3836 Pa  — exact match (4–5 sig figs across α∈[0.25,10])
   rmse       = 6.6e-3 Pa

Root cause history: before commit 0bd0f88f, the CPU PSTD power-law
absorption operator was a density-side per-axis correction multiplied by
Δt, missing the c² · ρ₀/Δt scaling factor and producing ~10¹¹× weaker
attenuation than k-Wave's pressure-side algebraic formulation. The fix
ports the GPU WGSL `absorb_pressure_correction` shader and k-wave-python
`kspace_solver.py:613` formulation to CPU: `p += c² · (τ·L1 − η·L2)` with
`L1 = IFFT(|k|^(y−2) · FFT(ρ₀·∇·u))`, `L2 = IFFT(|k|^(y−1) · FFT(ρ_total))`,
no Δt factor (algebraic, not integrated). The α-attenuation sweep
matches k-wave-python to 4–5 significant digits across α ∈ [0.25, 10.0]
dB/(MHz^y cm).

Remaining residual: pykwavers' peak fires exactly 1 dt earlier than
k-wave-python's at both sensors (independent of α — visible in the α=0
baseline). On a sub-Δt pulse the cross-correlation maximum drops below
0.97 even though the waveform shapes match. Tracked separately as a
propagator phase-offset audit; not blocking parity at the energy level.

Usage
-----
    python examples/na_modelling_absorption_compare.py
    python examples/na_modelling_absorption_compare.py --no-cache
    python examples/na_modelling_absorption_compare.py --allow-failure
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
# Physical constants (MATLAB case 2: alpha=0.25, y=1.5)
# ---------------------------------------------------------------------------
NX = 1024
DX = 12.8e-3 / NX
C0 = 1500.0
RHO0 = 1000.0
ALPHA_COEFF = 0.25  # dB/(MHz^y cm)
ALPHA_POWER = 1.5

T_END = 4e-6
SOURCE_POS = NX // 4  # 0-based
SOURCE_SENSOR_DIST = 0.5e-3  # m
SENSOR_SENSOR_DIST = 1e-3  # m

# ---------------------------------------------------------------------------
# Parity thresholds — 1-D case is the tightest dispersion test (no
# transverse spreading). Per the dispersion drift documented in
# ivp_loading_external_image_compare.py, expect Pearson ≥ 0.97 with
# RMS ratio in [0.85, 1.20] for short propagation distances (~10 mm)
# and a single pulse. PSNR ≥ 18 dB.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r": 0.97,
    "rms_ratio_min": 0.85,
    "rms_ratio_max": 1.20,
    "psnr_db": 18.0,
}

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "na_modelling_absorption_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_modelling_absorption_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "na_modelling_absorption_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "na_modelling_absorption_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


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
        }
    except Exception:
        return None


def _save_cache(
    path: os.PathLike, pressure: np.ndarray, nt: int, dt: float, runtime_s: float
) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        pressure=np.asarray(pressure, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


def build_shared_inputs() -> dict:
    """Build 1-D kgrid, p0 delta, and 2-sensor mask."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium

    kgrid = kWaveGrid(Vector([NX]), Vector([DX]))
    medium = kWaveMedium(
        sound_speed=C0,
        alpha_coeff=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )
    kgrid.makeTime(medium.sound_speed, 0.3, T_END)
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    # p0 delta at SOURCE_POS (0-based)
    p0 = np.zeros(NX, dtype=np.float64)
    p0[SOURCE_POS] = 1.0

    sensor_pos_1 = SOURCE_POS + int(round(SOURCE_SENSOR_DIST / DX))
    sensor_pos_2 = SOURCE_POS + int(
        round((SOURCE_SENSOR_DIST + SENSOR_SENSOR_DIST) / DX)
    )

    sensor_mask = np.zeros(NX, dtype=bool)
    sensor_mask[sensor_pos_1] = True
    sensor_mask[sensor_pos_2] = True

    d_cm = (sensor_pos_2 - sensor_pos_1) * DX * 100.0

    return {
        "kgrid": kgrid,
        "medium": medium,
        "p0": p0,
        "sensor_mask": sensor_mask,
        "sensor_pos_1": sensor_pos_1,
        "sensor_pos_2": sensor_pos_2,
        "d_cm": d_cm,
        "nt": nt,
        "dt": dt,
    }


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
    p0 = inputs["p0"]
    sensor_mask = inputs["sensor_mask"]
    nt = inputs["nt"]
    dt = inputs["dt"]

    source = kSource()
    source.p0 = p0

    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p"]

    print(f"  [k-wave] Running 1-D kspaceFirstOrder1D  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        smooth_p0=False,
        pml_inside=True,
        backend="python", device="cpu", quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result["p"], dtype=np.float64)
    n_sensors = int(sensor_mask.sum())
    if pressure.shape[0] != n_sensors and pressure.shape[1] == n_sensors:
        pressure = pressure.T
    if pressure.shape[0] != n_sensors:
        raise AssertionError(
            f"Unexpected k-wave shape {pressure.shape}; expected ({n_sensors}, *)"
        )

    _save_cache(_KWAVE_CACHE, pressure, nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    p0 = inputs["p0"]
    sensor_mask = inputs["sensor_mask"]
    nt = inputs["nt"]
    dt = inputs["dt"]

    # Quasi-1D in pykwavers: NY=NZ=1.
    grid = pkw.Grid(NX, 1, 1, DX, DX, DX)
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        absorption=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    p0_3d = p0[:, None, None].astype(np.float64)
    source = pkw.Source.from_initial_pressure(p0_3d)

    sensor_mask_3d = sensor_mask[:, None, None]
    sensor = pkw.Sensor.from_mask(sensor_mask_3d)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running CPU PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    pressure = np.asarray(result.sensor_data, dtype=np.float64)
    n_sensors = int(sensor_mask.sum())
    if pressure.shape[0] != n_sensors and pressure.shape[1] == n_sensors:
        pressure = pressure.T
    if pressure.shape[0] != n_sensors:
        raise AssertionError(
            f"Unexpected pykwavers shape {pressure.shape}; expected ({n_sensors}, *)"
        )

    _save_cache(_PKWAV_CACHE, pressure, nt, dt, runtime_s)
    return {"pressure": pressure, "nt": nt, "dt": dt, "runtime_s": runtime_s}


def plot_comparison(
    inputs: dict, kw: dict, pkw_res: dict, metrics: dict, *, status: str
) -> None:
    kw_p = kw["pressure"]
    py_p = pkw_res["pressure"]
    diff = py_p - kw_p
    nt = inputs["nt"]
    dt = inputs["dt"]
    t_axis = np.arange(nt) * dt * 1e6  # µs

    vmax = float(max(np.abs(kw_p).max(), np.abs(py_p).max(), 1e-30))
    dmax = float(max(np.abs(diff).max(), 1e-30))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # Panel 1: Sensor 1 trace overlay
    ax = axes[0, 0]
    ax.plot(t_axis, kw_p[0], "k-", lw=1.4, label="k-wave-python")
    ax.plot(t_axis, py_p[0], "r--", lw=1.0, label="pykwavers")
    ax.set_title(f"Sensor 1 @ x = {inputs['sensor_pos_1']} (d = {SOURCE_SENSOR_DIST*1e3:.1f} mm)")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Pressure [Pa]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Sensor 2 trace overlay
    ax = axes[0, 1]
    ax.plot(t_axis, kw_p[1], "k-", lw=1.4, label="k-wave-python")
    ax.plot(t_axis, py_p[1], "r--", lw=1.0, label="pykwavers")
    ax.set_title(
        f"Sensor 2 @ x = {inputs['sensor_pos_2']} "
        f"(d = {(SOURCE_SENSOR_DIST + SENSOR_SENSOR_DIST)*1e3:.1f} mm)"
    )
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Pressure [Pa]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Sensor matrices side by side
    ax = axes[1, 0]
    sensor_mat = np.vstack([kw_p, py_p])
    im = ax.imshow(
        sensor_mat, aspect="auto", origin="lower", cmap="seismic",
        vmin=-vmax, vmax=vmax,
        extent=[0, nt, 0, 4],
    )
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(["kw S1", "kw S2", "py S1", "py S2"])
    ax.set_xlabel("Time step")
    ax.set_title("Sensor matrices (k-wave above, pykwavers below)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 4: difference
    ax = axes[1, 1]
    im = ax.imshow(
        diff, aspect="auto", origin="lower", cmap="seismic",
        vmin=-dmax, vmax=dmax,
    )
    ax.set_title(f"diff (pykwavers − k-wave)  max|Δ|={dmax:.2e}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sensor index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"na_modelling_absorption (α={ALPHA_COEFF}, y={ALPHA_POWER}): "
        f"k-wave-python vs pykwavers  [{status}]\n"
        f"r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}  "
        f"PSNR={metrics['psnr_db']:.1f} dB",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(FIGURE_PATH), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Compare pykwavers with k-wave-python for "
                     "na_modelling_absorption (case 2 of 3).")
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force a fresh run.")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()
    no_cache = args.no_cache

    print("=" * 78)
    print("na_modelling_absorption (case 2): k-wave-python vs pykwavers")
    print(f"  Grid    : {NX} (1-D)  dx={DX*1e6:.2f} µm")
    print(f"  Medium  : c={C0} m/s  ρ={RHO0} kg/m³  "
          f"α={ALPHA_COEFF} dB/(MHz^y cm)  y={ALPHA_POWER}")
    print(f"  Source  : delta IVP at x_idx={SOURCE_POS}")
    print("  Sensor  : 2 binary points at offsets +0.5 mm and +1.5 mm")
    print("=" * 78)

    inputs = build_shared_inputs()
    print(f"  Nt={inputs['nt']}  dt={inputs['dt']:.3e} s  "
          f"sensor positions: {inputs['sensor_pos_1']}, {inputs['sensor_pos_2']}")

    print("\n[1/2] k-wave-python ...")
    kw = run_kwave(inputs, no_cache=no_cache)
    print(f"  shape={kw['pressure'].shape}  "
          f"peak={float(np.abs(kw['pressure']).max()):.3e} Pa")

    print("\n[2/2] pykwavers ...")
    pkw_res = run_pykwavers(inputs, no_cache=no_cache)
    print(f"  shape={pkw_res['pressure'].shape}  "
          f"peak={float(np.abs(pkw_res['pressure']).max()):.3e} Pa")

    print("\n--- Parity evaluation ---")
    metrics = compute_image_metrics(kw["pressure"], pkw_res["pressure"])
    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"] >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db": metrics["psnr_db"] >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  (target ≥ {thr['pearson_r']})  "
          f"{'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  psnr_db   : {metrics['psnr_db']:.2f} dB  (target ≥ {thr['psnr_db']} dB)  "
          f"{'OK' if checks['psnr_db'] else 'FAIL'}")
    print(f"  rmse      : {metrics['rmse']:.3e} Pa")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    plot_comparison(inputs, kw, pkw_res, metrics, status=status)

    header_lines = [
        "na_modelling_absorption (case 2) parity metrics",
        f"parity_status: {status}",
        f"grid: 1-D, NX={NX}  dx={DX:.6e} m",
        f"medium: c={C0}  rho={RHO0}  alpha={ALPHA_COEFF}  alpha_power={ALPHA_POWER}",
        f"source: delta IVP at x_idx={SOURCE_POS}",
        f"sensors: x={inputs['sensor_pos_1']} and x={inputs['sensor_pos_2']} (d={inputs['d_cm']:.3f} cm apart)",
        f"nt={inputs['nt']}  dt={inputs['dt']:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ]
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target ≥ {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f} dB  (target ≥ {thr['psnr_db']} dB)",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw['pressure']).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(pkw_res['pressure']).max()):.6e}",
        f"peak_ratio        = {metrics['peak_ratio']:.6f}",
    ]
    save_text_report(METRICS_PATH, "\n".join(header_lines), report_lines)
    print(f"  Saved: {METRICS_PATH}")
    return 0 if (status == "PASS" or args.allow_failure) else 1


if __name__ == "__main__":
    raise SystemExit(main())
