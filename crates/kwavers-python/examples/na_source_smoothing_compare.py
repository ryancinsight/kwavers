#!/usr/bin/env python3
"""
na_source_smoothing_compare.py
==============================
Parity comparison for the upstream ``na_source_smoothing`` example
(initial-pressure IVP with three frequency-domain window types).

Physical setup (matches k-wave-python ``examples/na_source_smoothing.py``):
  Grid:    1D — Nx = 256, dx = 0.05 mm  (pseudo-3D: 1 × 256 × 1 in pykwavers)
  Medium:  c = 1500 m/s, ρ = 1000 kg/m³, lossless
  Source:  delta-function p0 at grid centre (index Nx//2 - 1 = 127, 0-based)
  Window:  three cases — None, Hanning, Blackman (applied in k-space)
  Sensor:  single point ~2 μs propagation distance from source
  Time:    dt = 2 ns, t_end = 4.26 μs  (Nt = round(t_end/dt) + 1 = 2131)

Window theorem
--------------
A frequency-domain window w[k] applied to the k-space spectrum of the initial
pressure p̂₀ smooths the discrete delta to a bandlimited pulse, suppressing
Gibbs ringing.  After inverse FFT:

  p₀_smooth(x) = IFFT{ FFT{p₀}(k) · w(k) } / c_g

where c_g is the coherent gain of the window (so energy is conserved relative
to the rectangular window).

References
----------
* Treeby & Cox (2010), k-Wave MATLAB toolbox, ``smooth``.
* Harris (1978), "On the use of windows for harmonic analysis with the DFT,"
  Proc. IEEE 66(1):51–83.

Parity strategy
---------------
Both solvers receive the identical smoothed p0 field and identical lossless
medium.  The time-domain pressure trace at the sensor is compared with
Pearson correlation ≥ 0.95 and RMS ratio ∈ [0.80, 1.20] per window case.

Outputs
-------
* ``output/na_source_smoothing_compare.png``
* ``output/na_source_smoothing_metrics.txt``

Usage
-----
  python examples/na_source_smoothing_compare.py
  python examples/na_source_smoothing_compare.py --no-cache
  python examples/na_source_smoothing_compare.py --allow-failure
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
# Physical constants (must match na_source_smoothing.py exactly)
# ---------------------------------------------------------------------------
NX            = 256
DX            = 0.05e-3       # [m] — 0.05 mm
C0            = 1500.0        # [m/s]
RHO0          = 1000.0        # [kg/m³]
DT            = 2e-9          # [s]
T_END         = 4.26e-6       # [s]
NT            = int(np.round(T_END / DT)) + 1   # 2131

# Source: delta at centre (k-wave-python convention: Nx//2 - 1 = 127, 0-based)
SOURCE_IDX    = NX // 2 - 1   # 127

# Sensor: ~2 μs from source
_SOURCE_SENSOR_DIST = 2e-6    # [m]
_SENSOR_OFFSET = int(np.round(C0 * _SOURCE_SENSOR_DIST / DX))  # ≈ 60
SENSOR_IDX    = SOURCE_IDX + _SENSOR_OFFSET    # 187

WINDOW_CASES: list[tuple[str, str | None]] = [
    ("no_window", None),
    ("hanning",   "Hanning"),
    ("blackman",  "Blackman"),
]

# ---------------------------------------------------------------------------
# Parity thresholds
# ---------------------------------------------------------------------------
# The two solvers share the identical smoothed p0 and lossless medium.  The
# dominant discrepancy is numerical dispersion (k-space vs PSTD stencil order).
# Pearson ≥ 0.95 and RMS ratio ∈ [0.80, 1.20] are conservative but physically
# meaningful — a randomly signed trace would give |Pearson| ≈ 0.
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.95,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.20,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH   = DEFAULT_OUTPUT_DIR / "na_source_smoothing_compare.png"
METRICS_PATH  = DEFAULT_OUTPUT_DIR / "na_source_smoothing_metrics.txt"
_KWAVE_CACHE  = DEFAULT_OUTPUT_DIR / "na_source_smoothing_kwave_cache.npz"
_PKWAV_CACHE  = DEFAULT_OUTPUT_DIR / "na_source_smoothing_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Window application (matching k-wave-python `smooth` / `get_win` behaviour)
# ---------------------------------------------------------------------------
def _apply_window(p0: np.ndarray, window_type: str | None) -> np.ndarray:
    """Apply a symmetric frequency-domain window to a 1-D initial pressure.

    Theorem (Harris 1978): multiplying the DFT spectrum by a window w[k]
    and inverse-transforming gives a spatially-smoothed signal with peak
    sidelobe levels determined by the window family.  The coherent gain
    c_g = mean(w) normalises the total energy so the peak amplitude equals
    that of the original delta for arbitrary window types.

    Parameters
    ----------
    p0          : 1-D float64 array of length Nx.
    window_type : ``None`` for rectangular (no-op), ``'Hanning'`` for a
                  periodic Hanning window, or ``'Blackman'`` for a periodic
                  Blackman window.

    Returns
    -------
    Smoothed p0 (same shape as input, real-valued).
    """
    if window_type is None:
        return p0.copy()

    n = len(p0)
    if window_type == "Hanning":
        # Periodic Hanning: w[k] = 0.5 - 0.5 * cos(2π k / N)
        win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
    elif window_type == "Blackman":
        # Periodic Blackman: w[k] = 0.42 - 0.5 cos(2π k/N) + 0.08 cos(4π k/N)
        k = np.arange(n)
        win = (
            0.42
            - 0.5  * np.cos(2.0 * np.pi * k / n)
            + 0.08 * np.cos(4.0 * np.pi * k / n)
        )
    else:
        raise ValueError(f"Unknown window type: {window_type!r}")

    c_g = float(np.mean(win))   # coherent gain
    p0_k = np.fft.fftshift(np.fft.fft(p0))
    p0_smooth = np.real(np.fft.ifft(np.fft.ifftshift(p0_k * win))) / c_g
    return p0_smooth


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_kwave_cache() -> dict | None:
    if REFRESH_CACHE or not _KWAVE_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_KWAVE_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            k: {
                "trace":      np.asarray(d[f"{k}_trace"],     dtype=np.float64),
                "p0":         np.asarray(d[f"{k}_p0"],        dtype=np.float64),
                "runtime_s":  float(d[f"{k}_runtime_s"]),
            }
            for k, _ in WINDOW_CASES
        }
    except Exception:
        return None


def _save_kwave_cache(results: dict[str, dict]) -> None:
    _KWAVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int32),
    }
    for case_name, r in results.items():
        payload[f"{case_name}_trace"]     = np.asarray(r["trace"],     dtype=np.float64)
        payload[f"{case_name}_p0"]        = np.asarray(r["p0"],        dtype=np.float64)
        payload[f"{case_name}_runtime_s"] = np.array(r["runtime_s"],  dtype=np.float64)
    np.savez(os.fspath(_KWAVE_CACHE), **payload)


def _load_pkwav_cache() -> dict | None:
    if REFRESH_CACHE or not _PKWAV_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_PKWAV_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            k: {
                "trace":     np.asarray(d[f"{k}_trace"],     dtype=np.float64),
                "runtime_s": float(d[f"{k}_runtime_s"]),
            }
            for k, _ in WINDOW_CASES
        }
    except Exception:
        return None


def _save_pkwav_cache(results: dict[str, dict]) -> None:
    _PKWAV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int32),
    }
    for case_name, r in results.items():
        payload[f"{case_name}_trace"]     = np.asarray(r["trace"],     dtype=np.float64)
        payload[f"{case_name}_runtime_s"] = np.array(r["runtime_s"],  dtype=np.float64)
    np.savez(os.fspath(_PKWAV_CACHE), **payload)


# ---------------------------------------------------------------------------
# Step 1 — k-wave-python reference runs
# ---------------------------------------------------------------------------
def run_kwave(*, no_cache: bool = False) -> dict[str, dict]:
    """Run the upstream na_source_smoothing example for all three window cases.

    Returns
    -------
    dict mapping case name → {'trace': 1-D float64, 'p0': 1-D float64,
                               'runtime_s': float}
    """
    if not no_cache:
        cached = _load_kwave_cache()
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    results: dict[str, dict] = {}

    for case_name, window_type in WINDOW_CASES:
        # Build grid, medium, source (identical to upstream example)
        kgrid = kWaveGrid(Vector([NX]), Vector([DX]))
        kgrid.setTime(NT, DT)
        medium = kWaveMedium(sound_speed=C0)

        source = kSource()
        p0_delta = np.zeros(NX)
        p0_delta[SOURCE_IDX] = 1.0
        source.p0 = _apply_window(p0_delta, window_type)

        sensor_mask = np.zeros(NX, dtype=bool)
        sensor_mask[SENSOR_IDX] = True
        sensor = kSensor(mask=sensor_mask)
        sensor.record = ["p"]

        print(f"  [k-wave] Running case '{case_name}' (Nt={NT}, dt={DT:.2e} s)...")
        t0 = time.perf_counter()
        kw_result = kspaceFirstOrder(
            kgrid, medium, source, sensor,
            backend="python", device="cpu", quiet=True,
            pml_inside=True, smooth_p0=False,
        )
        runtime_s = time.perf_counter() - t0

        p_arr = np.asarray(kw_result["p"], dtype=np.float64).ravel()
        print(f"  [k-wave]   Done in {runtime_s:.1f} s — trace shape: {p_arr.shape}")
        results[case_name] = {
            "trace":     p_arr,
            "p0":        source.p0.ravel().astype(np.float64),
            "runtime_s": runtime_s,
        }

    _save_kwave_cache(results)
    return results


# ---------------------------------------------------------------------------
# Step 2 — pykwavers PSTD runs (pseudo-1D: 1 × Nx × 1 grid)
# ---------------------------------------------------------------------------
def run_pykwavers(
    kwave_results: dict[str, dict],
    *,
    no_cache: bool = False,
) -> dict[str, dict]:
    """Run pykwavers PSTD for all three window cases using the same p0 as k-wave.

    The identical windowed p0 array from the k-wave run is injected so the two
    solvers share the same initial condition exactly.

    Returns
    -------
    dict mapping case name → {'trace': 1-D float64, 'runtime_s': float}
    """
    if not no_cache:
        cached = _load_pkwav_cache()
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    results: dict[str, dict] = {}

    # Pseudo-1D grid: x-dimension = NX, y and z are singleton.
    grid = pkw.Grid(nx=NX, ny=1, nz=1, dx=DX, dy=DX, dz=DX)
    medium_obj = pkw.Medium(
        sound_speed=np.full((NX, 1, 1), C0),
        density=np.full((NX, 1, 1), RHO0),
    )

    # Sensor at (SENSOR_IDX, 0, 0) in grid space.
    sensor_mask = np.zeros((NX, 1, 1), dtype=bool)
    sensor_mask[SENSOR_IDX, 0, 0] = True
    sensor_obj = pkw.Sensor.from_mask(sensor_mask)

    for case_name, _ in WINDOW_CASES:
        # Reuse k-wave's windowed p0 so both solvers share identical IVP.
        p0_1d = kwave_results[case_name]["p0"]   # shape (Nx,)
        p0_3d = p0_1d.reshape(NX, 1, 1)

        source_obj = pkw.Source.from_initial_pressure(p0_3d)

        # k-wave's `kspaceFirstOrder` is the PSTD solver; use PSTD here so the
        # quasi-1D (ny = nz = 1) configuration is supported. Default FDTD
        # requires ny >= 2 because it uses staggered finite differences.
        sim = pkw.Simulation(
            grid, medium_obj, source_obj, sensor_obj,
            solver=pkw.SolverType.PSTD,
        )

        print(f"  [pykwavers] Running case '{case_name}' (Nt={NT}, dt={DT:.2e} s)...")
        t0 = time.perf_counter()
        result = sim.run(time_steps=NT, dt=DT)
        runtime_s = time.perf_counter() - t0

        # Sensor data: shape (n_sensors, Nt) or (Nt,).
        p_data = result.sensor_data
        if p_data is None:
            raise RuntimeError(f"pykwavers returned no sensor data for case '{case_name}'")
        p_data = np.asarray(p_data, dtype=np.float64)
        trace = p_data.ravel()
        print(f"  [pykwavers]   Done in {runtime_s:.1f} s — trace shape: {trace.shape}")
        results[case_name] = {
            "trace":     trace,
            "runtime_s": runtime_s,
        }

    _save_pkwav_cache(results)
    return results


# ---------------------------------------------------------------------------
# Step 3 — compare and report
# ---------------------------------------------------------------------------
def compare_and_report(
    kwave_results: dict[str, dict],
    pkwav_results: dict[str, dict],
) -> tuple[dict[str, dict], bool]:
    """Compute per-case metrics and evaluate parity gates.

    Returns
    -------
    metrics_by_case : dict mapping case name → metrics dict
    all_pass        : True iff all cases satisfy PARITY_THRESHOLDS
    """
    metrics_by_case: dict[str, dict] = {}
    all_pass = True

    for case_name, _ in WINDOW_CASES:
        ref   = kwave_results[case_name]["trace"]
        cand  = pkwav_results[case_name]["trace"]
        m     = compute_trace_metrics(ref, cand)

        pearson_ok  = m["pearson_r"]  >= PARITY_THRESHOLDS["pearson_r"]
        rms_ok      = (PARITY_THRESHOLDS["rms_ratio_min"]
                       <= m["rms_ratio"]
                       <= PARITY_THRESHOLDS["rms_ratio_max"])
        case_pass   = pearson_ok and rms_ok
        all_pass    = all_pass and case_pass

        m["case_pass"] = case_pass
        metrics_by_case[case_name] = m

        status = "PASS" if case_pass else "FAIL"
        print(
            f"  [{status}] {case_name:<12}  "
            f"Pearson={m['pearson_r']:.4f}  "
            f"RMS ratio={m['rms_ratio']:.4f}  "
            f"RMSE={m['rmse']:.4e}  "
            f"peak ratio={m['peak_ratio']:.4f}"
        )

    return metrics_by_case, all_pass


# ---------------------------------------------------------------------------
# Step 4 — plotting
# ---------------------------------------------------------------------------
def make_figure(
    kwave_results: dict[str, dict],
    pkwav_results: dict[str, dict],
    metrics_by_case: dict[str, dict],
) -> None:
    """Save a 6-panel comparison figure (time domain + spectrum per case)."""
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle("na_source_smoothing: k-wave-python vs pykwavers parity", fontsize=13)

    for row, (case_name, window_type) in enumerate(WINDOW_CASES):
        label = "No Window" if window_type is None else window_type

        ref   = kwave_results[case_name]["trace"]
        cand  = pkwav_results[case_name]["trace"]
        m     = metrics_by_case[case_name]
        status_str = "PASS" if m["case_pass"] else "FAIL"

        n = min(len(ref), len(cand))
        t_ax = np.arange(n) * DT * 1e6   # [μs]

        # Time-domain plot
        ax_t = axes[row, 0]
        ax_t.plot(t_ax[:n], ref[:n],  "b-",  lw=1.2, label="k-wave-python")
        ax_t.plot(t_ax[:n], cand[:n], "r--", lw=1.2, label="pykwavers",  alpha=0.85)
        ax_t.set_xlabel("Time [μs]")
        ax_t.set_ylabel("Pressure [Pa]")
        ax_t.set_title(
            f"{label}  [{status_str}]  "
            f"r={m['pearson_r']:.3f}  RMS ratio={m['rms_ratio']:.3f}"
        )
        ax_t.legend(fontsize=8)

        # Frequency-domain plot (single-sided amplitude spectrum)
        fs = 1.0 / DT
        ax_f = axes[row, 1]
        for sig, color, lab in [(ref, "b", "k-wave-python"), (cand, "r", "pykwavers")]:
            fft_amp = np.abs(np.fft.rfft(sig[:n])) / n
            fft_amp[1:-1] *= 2.0
            freqs = np.fft.rfftfreq(n, d=1.0 / fs) / 1e6   # [MHz]
            peak = fft_amp.max() + 1e-30
            ax_f.plot(freqs, fft_amp / peak, color=color, lw=1.2, label=lab, alpha=0.85)
        ax_f.set_xlabel("Frequency [MHz]")
        ax_f.set_ylabel("Normalised amplitude")
        ax_f.set_title(f"{label}: amplitude spectrum")
        ax_f.set_xlim(0, fs / 2e6)
        ax_f.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.fspath(FIGURE_PATH), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Step 5 — text report
# ---------------------------------------------------------------------------
def write_report(
    kwave_results: dict[str, dict],
    pkwav_results: dict[str, dict],
    metrics_by_case: dict[str, dict],
    all_pass: bool,
) -> None:
    lines: list[str] = [
        "na_source_smoothing_compare — parity report",
        "=" * 64,
        f"Nx={NX}  dx={DX*1e3:.4f} mm  C0={C0} m/s  RHO0={RHO0} kg/m^3",
        f"dt={DT*1e9:.2f} ns  Nt={NT}  t_end={T_END*1e6:.3f} us",
        f"Source index : {SOURCE_IDX}  Sensor index : {SENSOR_IDX}",
        "",
        "Parity thresholds:",
        f"  Pearson r     >= {PARITY_THRESHOLDS['pearson_r']:.2f}",
        f"  RMS ratio  in [{PARITY_THRESHOLDS['rms_ratio_min']:.2f}, "
        f"{PARITY_THRESHOLDS['rms_ratio_max']:.2f}]",
        "",
        "Per-case results:",
    ]

    for case_name, _ in WINDOW_CASES:
        m = metrics_by_case[case_name]
        kw_rt  = kwave_results[case_name]["runtime_s"]
        pkw_rt = pkwav_results[case_name]["runtime_s"]
        status = "PASS" if m["case_pass"] else "FAIL"
        lines += [
            f"  {case_name:<14} [{status}]",
            f"    Pearson r      = {m['pearson_r']:.6f}",
            f"    RMS ratio      = {m['rms_ratio']:.6f}",
            f"    RMSE           = {m['rmse']:.6e}",
            f"    peak ratio     = {m['peak_ratio']:.6f}",
            f"    k-wave runtime = {kw_rt:.2f} s",
            f"    pkwav runtime  = {pkw_rt:.2f} s",
            "",
        ]

    lines += [
        "-" * 64,
        f"OVERALL: {'PASS' if all_pass else 'FAIL'}",
    ]

    # save_text_report signature: (path, header, lines).
    save_text_report(METRICS_PATH, lines[0] if lines else "", lines[1:])
    print(f"  Report saved: {METRICS_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--no-cache",      action="store_true", help="Force re-run ignoring caches")
    p.add_argument("--allow-failure", action="store_true", help="Exit 0 even if parity gates fail")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    no_cache = args.no_cache or REFRESH_CACHE

    print("\n[ 1 / 4 ]  k-wave-python reference runs …")
    kwave_results = run_kwave(no_cache=no_cache)

    print("\n[ 2 / 4 ]  pykwavers PSTD runs …")
    pkwav_results = run_pykwavers(kwave_results, no_cache=no_cache)

    print("\n[ 3 / 4 ]  Comparing traces …")
    metrics, all_pass = compare_and_report(kwave_results, pkwav_results)

    print("\n[ 4 / 4 ]  Generating outputs …")
    make_figure(kwave_results, pkwav_results, metrics)
    write_report(kwave_results, pkwav_results, metrics, all_pass)

    overall = "PASS" if all_pass else "FAIL"
    print(f"\nOverall parity gate: {overall}")
    if not all_pass and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
