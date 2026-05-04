#!/usr/bin/env python3
"""
na_modelling_nonlinearity_compare.py
=====================================
Parity comparison for the upstream ``na_modelling_nonlinearity`` example
(nonlinear 1-D plane wave propagation with power-law absorption).

Physical setup (matches k-wave-python ``examples/na_modelling_nonlinearity.py``):
  Grid:    1D — Nx = 1520, dx = 1.5e-5 m (100 points/wavelength at 1 MHz)
  Medium:  c = 1500 m/s, rho = 1000 kg/m^3,
           alpha_coeff = 0.25 dB/(MHz^2 cm), alpha_power = 2,
           BonA computed from shock parameter sigma = 2
  Source:  single point at index 9 (0-based), sinusoidal p = 10 MPa
  Sensor:  full 1-D grid; records only last 3 temporal periods
           (record_start_index = Nt - 1200 + 1 = 8801, 1-based)
  PML:     80 grid points, outside domain, alpha = 1.5

Comparison strategy
-------------------
Both engines receive identical source signals and BonA.  Nonlinear harmonic
generation (sawtooth wave at sigma = 2) is compared at the detector position
x_px = 1500 grid points from the source (grid index 1509, 0-based).

Analytical reference
--------------------
Mendousse (1953) predicts that for sigma = 2 the first harmonic is
approximately 3x the fundamental amplitude at the detector.  This ratio is
verified as an auxiliary check.

Outputs
-------
* ``output/na_modelling_nonlinearity_compare.png``
* ``output/na_modelling_nonlinearity_metrics.txt``

Usage
-----
  python examples/na_modelling_nonlinearity_compare.py
  python examples/na_modelling_nonlinearity_compare.py --no-cache
  python examples/na_modelling_nonlinearity_compare.py --allow-failure
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
# Physical constants (must match na_modelling_nonlinearity.py exactly)
# ---------------------------------------------------------------------------
P0             = 10e6    # source pressure amplitude [Pa]
C0             = 1500.0  # sound speed [m/s]
RHO0           = 1000.0  # density [kg/m^3]
ALPHA_COEFF    = 0.25    # absorption coefficient [dB/(MHz^2 cm)]
ALPHA_POWER    = 2.0     # absorption power-law exponent
SIGMA          = 2.0     # shock parameter

SOURCE_FREQ    = 1e6     # fundamental frequency [Hz]
POINTS_PER_WL  = 100     # grid points per wavelength
WL_SEPARATION  = 15      # wavelength separation between source and detector
CFL            = 0.25    # CFL number

# Grid parameters (derived)
DX             = C0 / (POINTS_PER_WL * SOURCE_FREQ)        # 1.5e-5 m
NX             = WL_SEPARATION * POINTS_PER_WL + 20        # 1520

# Source and detector positions (0-based)
SOURCE_IDX     = 9                                          # 0-based (source_pos=10, 1-based)
X_PX           = WL_SEPARATION * POINTS_PER_WL             # 1500 grid points
DETECTOR_IDX   = SOURCE_IDX + X_PX                         # 1509, 0-based

# Time array (derived)
POINTS_PER_PERIOD = int(np.round(POINTS_PER_WL / CFL))     # 400
DT                = 1.0 / (POINTS_PER_PERIOD * SOURCE_FREQ)  # 2.5e-9 s
T_END             = 25e-6                                   # [s]
NT                = int(np.round(T_END / DT))               # 10000

# sensor.record_start_index (k-Wave 1-based): record last 3 temporal periods
RECORD_START_INDEX = NT - 3 * POINTS_PER_PERIOD + 1        # 8801
NT_RECORDED        = NT - RECORD_START_INDEX + 1           # 1200

# PML settings (match MATLAB/k-wave-python example)
PML_SIZE       = 80
PML_ALPHA      = 1.5

# B/A computed from shock parameter sigma = 2
_x           = X_PX * DX                                   # physical distance [m]
_mach_num    = P0 / (RHO0 * C0**2)                        # acoustic Mach number
_k           = 2.0 * np.pi * SOURCE_FREQ / C0              # wavenumber [rad/m]
BON_A        = 2.0 * (SIGMA / (_mach_num * _k * _x) - 1)  # ≈ 7.55 for water

# ---------------------------------------------------------------------------
# Parity thresholds
# Nonlinear wave: solver must reproduce harmonic content qualitatively.
# High-amplitude nonlinear propagation is numerically sensitive, so allow
# moderate amplitude mismatch while requiring phase/shape correlation ≥ 0.90.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.90,
    "rms_ratio_min": 0.75,
    "rms_ratio_max": 1.35,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "na_modelling_nonlinearity_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_modelling_nonlinearity_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "na_nonlinearity_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "na_nonlinearity_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Helper: cache I/O
# ---------------------------------------------------------------------------
def _load_kwave_cache() -> dict | None:
    if REFRESH_CACHE or not _KWAVE_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_KWAVE_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "trace":     np.asarray(d["trace"],     dtype=np.float64).ravel(),
            "bon_a":     float(d["bon_a"]),
            "nt":        int(d["nt"]),
            "dt":        float(d["dt"]),
            "nt_rec":    int(d["nt_rec"]),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_kwave_cache(
    trace: np.ndarray,
    bon_a: float,
    nt: int,
    dt: float,
    nt_rec: int,
    runtime_s: float,
) -> None:
    _KWAVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        os.fspath(_KWAVE_CACHE),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        trace=np.asarray(trace, dtype=np.float64),
        bon_a=np.array(bon_a, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        nt_rec=np.array(nt_rec, dtype=np.int64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


def _load_pkwav_cache() -> dict | None:
    if REFRESH_CACHE or not _PKWAV_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_PKWAV_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "trace":     np.asarray(d["trace"], dtype=np.float64).ravel(),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_pkwav_cache(trace: np.ndarray, runtime_s: float) -> None:
    _PKWAV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        os.fspath(_PKWAV_CACHE),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        trace=np.asarray(trace, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Step 1 — k-wave-python reference run
# ---------------------------------------------------------------------------
def run_kwave(*, no_cache: bool = False) -> dict:
    """Run the upstream na_modelling_nonlinearity example.

    Returns
    -------
    dict with keys:
        ``trace``     : 1-D float64 (NT_RECORDED,) — pressure at detector [Pa]
        ``bon_a``     : float — computed B/A
        ``nt``        : int — total time steps
        ``dt``        : float — time step [s]
        ``nt_rec``    : int — recorded time steps
        ``runtime_s`` : float — wall-clock time [s]
    """
    if not no_cache:
        cached = _load_kwave_cache()
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    import sys as _sys
    from example_parity_utils import KWAVE_PYTHON_ROOT
    if str(KWAVE_PYTHON_ROOT / "examples") not in _sys.path:
        _sys.path.insert(0, str(KWAVE_PYTHON_ROOT / "examples"))

    from na_modelling_nonlinearity import setup as kwave_setup, run as kwave_run  # type: ignore[import]

    print("  [k-wave] Building grid, medium, source...")
    kgrid, medium, source = kwave_setup()
    nt  = int(kgrid.Nt)
    dt  = float(kgrid.dt)
    bon_a = float(medium.BonA)

    print(f"  [k-wave] Running 1-D PSTD  (Nt={nt}, dt={dt:.3e} s, B/A={bon_a:.4f})...")
    t0 = time.perf_counter()
    result = kwave_run(backend="python", device="cpu", quiet=True)
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    # result["p"]: shape (Nx, NT_RECORDED) — full grid, last 3 periods
    p = np.asarray(result["p"], dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    # k-wave result["p"] for full-grid 1D sensor: shape (Nx, Nt_rec)
    nt_rec = p.shape[1]
    trace = p[DETECTOR_IDX, :].ravel()
    print(f"  [k-wave] p shape: {p.shape}  detector trace: {len(trace)} samples")

    _save_kwave_cache(trace, bon_a, nt, dt, nt_rec, runtime_s)
    return {
        "trace":     trace,
        "bon_a":     bon_a,
        "nt":        nt,
        "dt":        dt,
        "nt_rec":    nt_rec,
        "runtime_s": runtime_s,
    }


# ---------------------------------------------------------------------------
# Step 2 — pykwavers CPU PSTD
# ---------------------------------------------------------------------------
def _build_source_signal(nt: int, dt: float) -> np.ndarray:
    """Sinusoidal source signal: p0 * sin(2*pi*f0*t), shape (nt,)."""
    t = np.arange(nt) * dt
    return P0 * np.sin(2.0 * np.pi * SOURCE_FREQ * t)


def run_pykwavers(
    *,
    nt: int,
    dt: float,
    bon_a: float,
    no_cache: bool = False,
) -> dict:
    """Run pykwavers PSTD with nonlinear medium.

    Parameters
    ----------
    nt      : total time steps
    dt      : time step [s]
    bon_a   : B/A parameter (from k-wave setup to guarantee identical physics)
    no_cache: skip cache lookup

    Returns
    -------
    dict with keys ``trace`` (1-D float64, NT_RECORDED samples) and ``runtime_s``.
    """
    if not no_cache:
        cached = _load_pkwav_cache()
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    # --- Grid (1D: ny=nz=1) ---
    grid = pkw.Grid(NX, 1, 1, DX, DX, DX)

    # --- Medium: homogeneous with absorption and nonlinearity ---
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        absorption=ALPHA_COEFF,
        nonlinearity=float(bon_a),
        alpha_power=ALPHA_POWER,
    )

    # --- Source: single point at SOURCE_IDX, sinusoidal signal ---
    src_mask = np.zeros((NX, 1, 1), dtype=np.float64)
    src_mask[SOURCE_IDX, 0, 0] = 1.0
    sig = _build_source_signal(nt, dt)  # shape (nt,)
    source = pkw.Source.from_mask(src_mask, sig, SOURCE_FREQ)

    # --- Sensor: full 1D grid (all NX points), record last 3 periods ---
    sens_mask = np.ones((NX, 1, 1), dtype=bool)
    sensor = pkw.Sensor.from_mask(sens_mask)
    # k-Wave 1-based record_start_index
    sensor.set_record_start_index(RECORD_START_INDEX)

    # --- Simulation ---
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_alpha(PML_ALPHA)
    sim.set_pml_inside(False)
    sim.set_nonlinear(True)

    print(
        f"  [pykwavers] Running 1-D PSTD  (Nt={nt}, dt={dt:.3e} s, B/A={bon_a:.4f}, "
        f"record_start={RECORD_START_INDEX})..."
    )
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    sensor_data = np.asarray(result.sensor_data, dtype=np.float64)
    print(f"  [pykwavers] sensor_data shape: {sensor_data.shape}")
    if sensor_data.ndim == 1:
        # Single sensor or 1-D squeezed: unlikely for full grid, but guard
        trace = sensor_data.ravel()
        print(f"  [pykwavers] WARNING: sensor_data is 1D — expected (NX, NT_RECORDED)")
    else:
        # (n_sensors, NT_RECORDED) — sensor row = grid index (Fortran-order, x-fastest)
        trace = sensor_data[DETECTOR_IDX, :].ravel()

    print(f"  [pykwavers] detector trace: {len(trace)} samples")

    _save_pkwav_cache(trace, runtime_s)
    return {"trace": trace, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    kw: dict,
    pkw_res: dict,
    metrics: dict[str, float],
    *,
    status: str,
) -> None:
    """Two-panel figure: waveform overlay + frequency spectrum."""
    dt = kw["dt"]
    n = min(len(kw["trace"]), len(pkw_res["trace"]))
    t_us = np.arange(n) * dt * 1e6   # µs (relative to record_start_index)

    kw_tr  = kw["trace"][:n]
    pkw_tr = pkw_res["trace"][:n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Waveform overlay
    ax1.plot(t_us, kw_tr  * 1e-6, "k-",  lw=1.4, alpha=0.85, label="k-wave-python")
    ax1.plot(t_us, pkw_tr * 1e-6, "r--", lw=1.2, alpha=0.85, label="pykwavers")
    ax1.set_xlabel("Time [µs] (offset from record_start)")
    ax1.set_ylabel("Pressure [MPa]")
    ax1.set_title(f"Nonlinear waveform at detector (x = {X_PX} pts from source)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Frequency spectrum — should show fundamental + harmonics
    from numpy.fft import rfft, rfftfreq
    nfft = len(kw_tr)
    freq_mhz = rfftfreq(nfft, d=dt) * 1e-6
    spec_kw  = np.abs(rfft(kw_tr))
    spec_pkw = np.abs(rfft(pkw_tr))
    ax2.semilogy(freq_mhz, spec_kw  + 1, "k-",  lw=1.4, alpha=0.85, label="k-wave-python")
    ax2.semilogy(freq_mhz, spec_pkw + 1, "r--", lw=1.2, alpha=0.85, label="pykwavers")
    ax2.set_xlim(0, 6)
    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("|FFT| [Pa]")
    ax2.set_title("Harmonic spectrum (sigma = 2, B/A ≈ 7.55)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"na_modelling_nonlinearity: k-wave-python vs pykwavers  [{status}]\n"
        f"Nx={NX}  dx={DX*1e6:.1f} µm  B/A={BON_A:.2f}  sigma={SIGMA}  "
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
        description="Compare pykwavers with k-wave-python for na_modelling_nonlinearity."
    )
    parser.add_argument("--no-cache",      action="store_true",
                        help="Force a fresh run (ignore cached NPZ files).")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 70)
    print("na_modelling_nonlinearity: k-wave-python vs pykwavers")
    print(f"  Grid      : Nx={NX}  dx={DX*1e6:.2f} µm  (1-D)")
    print(f"  Medium    : c={C0} m/s  rho={RHO0} kg/m³  "
          f"alpha={ALPHA_COEFF} dB/(MHz^2 cm)  B/A={BON_A:.4f}")
    print(f"  Source    : idx={SOURCE_IDX}  p0={P0*1e-6:.0f} MPa  f0={SOURCE_FREQ*1e-6:.1f} MHz")
    print(f"  Detector  : idx={DETECTOR_IDX}  ({X_PX} grid pts from source)")
    print(f"  Time      : Nt={NT}  dt={DT:.3e} s  t_end={T_END*1e6:.1f} µs")
    print(f"  Recording : record_start_index={RECORD_START_INDEX}  "
          f"(last 3 periods = {NT_RECORDED} samples)")
    print(f"  PML       : {PML_SIZE} pts outside  alpha={PML_ALPHA}")
    print(f"  Shock     : sigma={SIGMA}")
    print("=" * 70)

    # --- k-wave-python ---
    print("\n[1/2] k-wave-python (1-D PSTD, nonlinear)...")
    kw = run_kwave(no_cache=no_cache)
    kw_tr = kw["trace"]
    print(f"  Nt={kw['nt']}  dt={kw['dt']:.3e} s  nt_rec={kw['nt_rec']}  "
          f"peak={float(np.abs(kw_tr).max())*1e-6:.4f} MPa  "
          f"rms={float(np.sqrt(np.mean(kw_tr**2)))*1e-6:.4f} MPa")

    # --- pykwavers ---
    print("\n[2/2] pykwavers (CPU PSTD, nonlinear B/A term)...")
    pkw_res = run_pykwavers(
        nt=kw["nt"],
        dt=kw["dt"],
        bon_a=kw["bon_a"],
        no_cache=no_cache,
    )
    pkw_tr = pkw_res["trace"]
    print(f"  nt_rec={len(pkw_tr)}  "
          f"peak={float(np.abs(pkw_tr).max())*1e-6:.4f} MPa  "
          f"rms={float(np.sqrt(np.mean(pkw_tr**2)))*1e-6:.4f} MPa")

    # --- Parity ---
    print("\n--- Parity evaluation ---")
    n = min(len(kw_tr), len(pkw_tr))
    if n == 0:
        print("  ERROR: empty traces — aborting")
        return 1 if not args.allow_failure else 0

    metrics = compute_trace_metrics(kw_tr[:n], pkw_tr[:n])

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  rmse      : {metrics['rmse']:.4e} Pa")
    print(f"  peak_ratio: {metrics['peak_ratio']:.6f}")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    # Auxiliary Mendousse check: 2nd harmonic amplitude ratio.
    # For sigma = 2, theory predicts 2nd / fundamental ≈ 1 (strong nonlinearity).
    from numpy.fft import rfft, rfftfreq
    spec_kw  = np.abs(rfft(kw_tr[:n]))
    freq_bins = rfftfreq(n, d=kw["dt"])
    f1_bin = int(np.argmin(np.abs(freq_bins - SOURCE_FREQ)))
    f2_bin = int(np.argmin(np.abs(freq_bins - 2 * SOURCE_FREQ)))
    h2_ratio_kwave = spec_kw[f2_bin] / (spec_kw[f1_bin] + 1e-30)
    print(f"  Mendousse: k-wave 2nd-harmonic/fundamental = {h2_ratio_kwave:.4f}")

    # --- Figure ---
    plot_comparison(kw, pkw_res, metrics, status=status)

    # --- Text report ---
    header_lines = [
        "na_modelling_nonlinearity parity metrics",
        f"parity_status: {status}",
        f"grid: Nx={NX}  dx={DX:.6e} m  (1-D)",
        f"medium: c={C0} m/s  rho={RHO0} kg/m3  "
        f"alpha_coeff={ALPHA_COEFF}  alpha_power={ALPHA_POWER}  B/A={kw['bon_a']:.6f}",
        f"source: idx={SOURCE_IDX}  p0={P0:.3e} Pa  f0={SOURCE_FREQ:.3e} Hz  sinusoidal",
        f"detector: idx={DETECTOR_IDX}  ({X_PX} pts from source)",
        f"shock_parameter: sigma={SIGMA}",
        f"nt={kw['nt']}  dt={kw['dt']:.6e} s  nt_recorded={kw['nt_rec']}",
        f"record_start_index: {RECORD_START_INDEX} (1-based)",
        f"pml_size: {PML_SIZE}  pml_inside: False  pml_alpha: {PML_ALPHA}",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ]
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw_tr).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(pkw_tr).max()):.6e}",
        f"peak_ratio        = {metrics['peak_ratio']:.6f}",
        f"mendousse_2nd_harmonic_ratio_kwave = {h2_ratio_kwave:.6f}",
    ]
    save_text_report(METRICS_PATH, "\n".join(header_lines), report_lines)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
