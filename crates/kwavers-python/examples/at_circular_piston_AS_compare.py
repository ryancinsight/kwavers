#!/usr/bin/env python3
"""
at_circular_piston_AS_compare.py
=================================
Side-by-side comparison of k-wave-python (`kspaceFirstOrderASC`) vs pykwavers
PSTD axisymmetric solver for the circular piston transducer example.

Physical setup (matches k-wave-python `at_circular_piston_AS.py`):
    Grid:      (Nx, Ny) AS — Nx = axial, Ny = radial (Nr)
    Medium:    homogeneous, c0 = 1500 m/s, rho = 1000 kg/m³, lossless
    Source:    line element (WSWA-FFT BLI), diam = 10 mm
               driven at 1 MHz CW, 1 MPa amplitude
    Sensor:    full 2-D mask (all grid points)
    PML:       outside (pml_inside=False)
    t_end:     40 µs (enough to reach steady state at 1 MHz)

Analytical reference:
    Pierce (1989) Eq 5-7.3 (circular piston in infinite baffle):
        p(x) = source_mag * |2 sin((k sqrt(x² + a²) - k x) / 2)|
    where a = source_diam / 2, k = 2π f0 / c0.

Validation:
    On-axis amplitude profile (steady-state) compared between k-wave reference,
    pykwavers AS PSTD, and Pierce analytical solution.

Usage:
    python examples/at_circular_piston_AS_compare.py
    python examples/at_circular_piston_AS_compare.py --no-cache
    python examples/at_circular_piston_AS_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrderAS import kspaceFirstOrderASC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.filters import extract_amp_phase
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.math import round_even
from kwave.utils.signals import create_cw_signals

# ---------------------------------------------------------------------------
# Physical parameters (from at_circular_piston_AS.py)
# ---------------------------------------------------------------------------
C0            = 1500.0          # [m/s]
RHO0          = 1000.0          # [kg/m³]
SOURCE_F0     = 1.0e6           # [Hz]
SOURCE_DIAM   = 10e-3           # piston diameter [m]
SOURCE_AMP    = np.array([1.0e6])  # [Pa]
SOURCE_PHASE  = np.array([0.0])    # [rad]
AXIAL_SIZE    = 32e-3           # total axial extent [m]
LATERAL_SIZE  = 8e-3            # total radial extent [m]
PPW           = 4               # points per wavelength
T_END         = 40e-6           # [s]
RECORD_PERIODS = 1
CFL           = 0.05
BLI_TOL       = 0.05
UPSAMPLE_RATE = 10

# Grid spacing
DX = C0 / (PPW * SOURCE_F0)     # [m] = 375 µm

# Grid sizes (k-wave convention: 2D Nx×Ny, axial×radial)
NX = round_even(AXIAL_SIZE / DX)
NY = round_even(LATERAL_SIZE / DX)

# Time discretisation
PPP = round(PPW / CFL)
DT  = 1.0 / (PPP * SOURCE_F0)
NT  = round(T_END / DT)

# Parity thresholds: pykwavers vs k-wave on-axis steady-state amplitude
PARITY_THRESHOLDS = {
    "pearson_r":     0.90,
    "rms_ratio_min": 0.75,
    "rms_ratio_max": 1.35,
    "psnr_db":       12.0,
}
ANALYTICAL_REFERENCE_THRESHOLDS = {
    "kwave_pearson_min": 0.99,
    "pykwavers_pearson_min": 0.98,
    "pearson_agreement_abs_max": 5e-6,
}

# Paths
FIGURE_PATH       = DEFAULT_OUTPUT_DIR / "at_circular_piston_AS_compare.png"
METRICS_PATH      = DEFAULT_OUTPUT_DIR / "at_circular_piston_AS_metrics.txt"
KWAVE_CACHE       = DEFAULT_OUTPUT_DIR / "at_circular_piston_AS_kwave_cache.npz"
KWAVE_RAW_CACHE   = DEFAULT_OUTPUT_DIR / "at_circular_piston_AS_kwave_raw.npz"
PKWAV_CACHE       = DEFAULT_OUTPUT_DIR / "at_circular_piston_AS_pykwavers_cache.npz"
PKWAV_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Build k-wave grid + source (shared setup reused by both engines)
# ---------------------------------------------------------------------------
def _build_kwave_grid():
    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DX]))
    kgrid.setTime(NT, DT)
    return kgrid


def _build_kwave_source(kgrid):
    t_arr = np.squeeze(kgrid.t_array)
    source_sig = create_cw_signals(t_arr, SOURCE_F0, SOURCE_AMP, SOURCE_PHASE)

    karray = KWaveArray_Kwave(
        axisymmetric=True,
        bli_tolerance=BLI_TOL,
        upsampling_rate=UPSAMPLE_RATE,
        single_precision=True,
    )
    # Circular piston: line element spanning radial half-aperture [0, diam/2]
    # at the first axial grid position (x = x_vec[0])
    x0 = kgrid.x_vec[0].item()
    karray.add_line_element([x0, -SOURCE_DIAM / 2.0], [x0, SOURCE_DIAM / 2.0])

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)        # (NX, NY) bool
    source.p      = karray.get_distributed_source_signal(kgrid, source_sig)  # (n_active, NT)
    return source, karray


# ---------------------------------------------------------------------------
# Run k-wave-python (reference)
# ---------------------------------------------------------------------------
def run_kwave() -> dict:
    if KWAVE_CACHE.exists():
        print("  [k-wave] Loading from cache...")
        d = np.load(KWAVE_CACHE)
        return {"amp_on_axis": d["amp_on_axis"], "amp_2d": d["amp_2d"], "runtime_s": float(d["runtime_s"])}

    # Check if raw sensor data was cached from a previous (interrupted) run
    if KWAVE_RAW_CACHE.exists():
        print("  [k-wave] Reprocessing from raw sensor cache (skipping k-wave binary)...")
        d = np.load(KWAVE_RAW_CACHE)
        p_raw = d["p_raw"]
        elapsed = float(d["runtime_s"])
    else:
        kgrid  = _build_kwave_grid()
        source, _ = _build_kwave_source(kgrid)
        medium = kWaveMedium(sound_speed=C0, density=RHO0)

        sensor = kSensor()
        sensor.mask = np.ones((NX, NY), dtype=bool)
        sensor.record = ["p"]
        sensor.record_start_index = kgrid.Nt - RECORD_PERIODS * PPP + 1

        sim_opts = SimulationOptions(
            simulation_type=SimulationType.AXISYMMETRIC,
            pml_inside=False,
            pml_auto=True,
            save_to_disk=True,
            save_to_disk_exit=False,
        )
        exec_opts = SimulationExecutionOptions(is_gpu_simulation=False, delete_data=False, verbose_level=2)

        print("  [k-wave] Running kspaceFirstOrderASC...")
        t0 = time.perf_counter()
        sensor_data = kspaceFirstOrderASC(
            medium=medium, kgrid=kgrid, source=source, sensor=sensor,
            simulation_options=sim_opts, execution_options=exec_opts,
        )
        elapsed = time.perf_counter() - t0
        print(f"  [k-wave] Done in {elapsed:.1f} s")

        p_raw = np.asarray(sensor_data["p"], dtype=np.float64)
        # Save raw sensor data immediately to avoid re-running the slow binary
        np.savez(KWAVE_RAW_CACHE, p_raw=p_raw, runtime_s=elapsed)

    # k-wave returns sensor_data["p"] as (Nt_recorded, n_sensor).
    # Transpose to (n_sensor, Nt_recorded) so extract_amp_phase dim=1 acts on time axis.
    amp, _, _ = extract_amp_phase(p_raw.T, 1.0 / DT, SOURCE_F0, dim=1, fft_padding=1, window="Rectangular")
    amp = np.asarray(amp, dtype=np.float64).flatten()

    # Reshape: full (NX, NY) grid in k-wave Fortran column-major order
    amp_2d = np.reshape(amp, (NX, NY), order="F")
    amp_on_axis = amp_2d[:, 0]

    result = {"amp_on_axis": amp_on_axis, "amp_2d": amp_2d, "runtime_s": elapsed}
    np.savez(KWAVE_CACHE, **result)
    return result


# ---------------------------------------------------------------------------
# Run pykwavers (AS PSTD)
# ---------------------------------------------------------------------------
def run_pykwavers() -> dict:
    if PKWAV_CACHE.exists():
        print("  [pykwavers] Loading from cache...")
        d = np.load(PKWAV_CACHE)
        cache_version = int(np.asarray(d["cache_version"]).reshape(())) if "cache_version" in d.files else 0
        if cache_version == PKWAV_CACHE_VERSION:
            return {
                "amp_on_axis": d["amp_on_axis"],
                "amp_2d": d["amp_2d"],
                "runtime_s": float(d["runtime_s"]),
            }

    # Use k-wave-python to generate the BLI source mask and distributed signal.
    kgrid  = _build_kwave_grid()
    source, _ = _build_kwave_source(kgrid)

    p_mask_2d = np.asarray(source.p_mask, dtype=np.float64)   # (NX, NY) float weights
    p_sig_2d  = np.asarray(source.p,      dtype=np.float64)   # (n_active, NT)

    # pykwavers grid: (NX, 1, NY) — axial × trivial × radial
    grid   = pkw.Grid(NX, 1, NY, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    # Reshape mask to (NX, 1, NY); active-cell ordering matches when ny=1
    p_mask_3d = p_mask_2d[:, np.newaxis, :]   # (NX, 1, NY)

    pkw_source = pkw.Source.from_mask(p_mask_3d, p_sig_2d, SOURCE_F0, mode="additive")

    # Sensor: full 3D mask (all grid points)
    sensor_mask = np.ones((NX, 1, NY), dtype=bool)
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, pkw_source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_axisymmetric(True)
    sim.set_pml_inside(False)

    print("  [pykwavers] Running PSTD AS...")
    t0 = time.perf_counter()
    result_run = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    p = np.asarray(result_run.sensor_data, dtype=np.float64)   # (n_sensor, NT)
    n_record = RECORD_PERIODS * PPP
    p_ss = p[:, -n_record:]

    amp, _, _ = extract_amp_phase(p_ss, 1.0 / DT, SOURCE_F0, dim=1, fft_padding=1, window="Rectangular")
    amp = np.asarray(amp, dtype=np.float64).flatten()

    # Sensor recorder emits Fortran-order rows to match MATLAB / k-Wave `find`.
    # Reshape in the same order before selecting the on-axis radial slice.
    amp_2d = amp.reshape(NX, NY, order="F")
    amp_on_axis = amp_2d[:, 0]

    output = {"amp_on_axis": amp_on_axis, "amp_2d": amp_2d, "runtime_s": elapsed}
    np.savez(PKWAV_CACHE, cache_version=PKWAV_CACHE_VERSION, **output)
    return output


# ---------------------------------------------------------------------------
# Pierce analytical reference (Eq 5-7.3)
# ---------------------------------------------------------------------------
def analytical_pierce(x_vec_m: np.ndarray) -> np.ndarray:
    """
    Pierce (1989) Eq 5-7.3: on-axis pressure amplitude for circular piston.

    p(x) = source_mag * |2 sin((k sqrt(x² + a²) - k x) / 2)|

    Parameters
    ----------
    x_vec_m : 1-D array of axial positions [m] measured from the piston face.
    """
    a = SOURCE_DIAM / 2.0
    k = 2.0 * np.pi * SOURCE_F0 / C0
    r = np.sqrt(x_vec_m**2 + a**2)
    return float(SOURCE_AMP[0]) * np.abs(2.0 * np.sin((k * r - k * x_vec_m) / 2.0))


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    if args.no_cache:
        for p in [KWAVE_CACHE, KWAVE_RAW_CACHE, PKWAV_CACHE]:
            if p.exists():
                p.unlink()
                print(f"  Removed cache: {p}")

    print("=" * 60)
    print("at_circular_piston_AS: k-wave-python vs pykwavers")
    print(f"  Grid     : {NX}×{NY} (axial×radial)  dx={DX*1e3:.3f} mm")
    print(f"  Medium   : c0={C0} m/s  rho={RHO0} kg/m³  lossless")
    print(f"  Source   : line element (BLI piston)  diam={SOURCE_DIAM*1e3:.0f} mm")
    print(f"             amp={SOURCE_AMP[0]*1e-6:.1f} MPa  f0={SOURCE_F0*1e-6:.1f} MHz")
    print(f"  Time     : dt={DT*1e9:.1f} ns  Nt={NT}  t_end={T_END*1e6:.0f} µs")
    print(f"  PML      : outside")
    print("=" * 60)

    print("\n[1/2] Running k-wave reference...")
    kw = run_kwave()

    print("\n[2/2] Running pykwavers AS PSTD...")
    pw = run_pykwavers()

    kw_amp = kw["amp_on_axis"]
    pw_amp = pw["amp_on_axis"]

    if kw_amp.shape != pw_amp.shape:
        raise AssertionError(
            f"on-axis amplitude shape mismatch: k-wave {kw_amp.shape} vs pykwavers {pw_amp.shape}"
        )

    metrics = compute_image_metrics(kw_amp, pw_amp)

    # On-axis positions (from grid origin = piston face)
    x_vec_m = np.arange(NX) * DX

    # Analytical Pierce reference evaluated at grid points
    p_ref = analytical_pierce(x_vec_m)

    # Pearson correlations vs analytical
    # Exclude x=0 (sin → 0, division not meaningful) — start from index 1
    r_kw_vs_ref, _ = pearsonr(p_ref[1:], kw_amp[1:])
    r_pkw_vs_ref, _ = pearsonr(p_ref[1:], pw_amp[1:])

    print("\n--- Parity evaluation (pykwavers vs k-wave reference) ---")
    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r":  metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio":  thr["rms_ratio_min"]  <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":    metrics["psnr_db"]    >= thr["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status     : {status}")
    print(f"  Pearson r  = {metrics['pearson_r']:.4f}  (≥ {thr['pearson_r']})")
    print(f"  RMS ratio  = {metrics['rms_ratio']:.4f}  "
          f"([{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])")
    print(f"  PSNR       = {metrics['psnr_db']:.2f} dB  (≥ {thr['psnr_db']} dB)")
    print(f"\n--- vs Pierce analytical (x > 0) ---")
    analytical_thr = ANALYTICAL_REFERENCE_THRESHOLDS
    print(
        "  k-wave vs analytical    pearson r = "
        f"{r_kw_vs_ref:.4f}  (>= {analytical_thr['kwave_pearson_min']})"
    )
    print(
        "  pykwavers vs analytical pearson r = "
        f"{r_pkw_vs_ref:.4f}  (>= {analytical_thr['pykwavers_pearson_min']})"
    )
    print(f"\n  k-wave runtime     : {kw['runtime_s']:.1f} s")
    print(f"  pykwavers runtime  : {pw['runtime_s']:.1f} s")

    # Dense analytical curve for plotting
    x_ref_m = np.linspace(0.0, (NX - 1) * DX, 10000)
    p_ref_dense = analytical_pierce(x_ref_m)

    x_mm = x_vec_m * 1e3
    x_ref_mm = x_ref_m * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: on-axis pressure comparison
    ax = axes[0]
    ax.plot(x_ref_mm, p_ref_dense * 1e-6, "k-", lw=1.4, label="Pierce analytical")
    ax.plot(x_mm, kw_amp * 1e-6, "b-", lw=1.6, alpha=0.8, label="k-wave (kspaceFirstOrderASC)")
    ax.plot(x_mm, pw_amp * 1e-6, "r--", lw=1.4, alpha=0.85, label="pykwavers (WSWA-FFT AS)")
    ax.set_xlabel("Axial position [mm]")
    ax.set_ylabel("Steady-state pressure amplitude [MPa]")
    ax.set_title(
        f"at_circular_piston_AS: on-axis amplitude  |  {status}\n"
        f"diam={SOURCE_DIAM*1e3:.0f} mm  f0={SOURCE_F0*1e-6:.1f} MHz  "
        f"{NX}×{NY} grid  dx={DX*1e6:.0f} µm"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: 2D pressure maps (k-wave vs pykwavers)
    kw_amp_2d = kw.get("amp_2d", None)
    pw_amp_2d = pw.get("amp_2d", None)
    if kw_amp_2d is not None and pw_amp_2d is not None:
        # Reflect radial half-plane to full plane for visualization
        y_mm = np.arange(NY) * DX * 1e3
        diff = pw_amp_2d - kw_amp_2d
        vmax = float(np.percentile(np.abs(diff), 99))
        im = axes[1].imshow(
            diff.T,
            aspect="auto",
            origin="lower",
            extent=[0, (NX - 1) * DX * 1e3, 0, (NY - 1) * DX * 1e3],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(im, ax=axes[1], label="ΔP [Pa]")
        axes[1].set_xlabel("Axial position [mm]")
        axes[1].set_ylabel("Radial position [mm]")
        axes[1].set_title("pykwavers − k-wave pressure amplitude [Pa]")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved figure : {FIGURE_PATH}")

    header = "\n".join([
        "at_circular_piston_AS parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY} (axial x radial)  dx={DX:.6e} m",
        f"source: line element (piston)  diam={SOURCE_DIAM} m  amp={SOURCE_AMP[0]} Pa  f0={SOURCE_F0} Hz",
        f"dt={DT:.6e} s  Nt={NT}  t_end={T_END} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pw['runtime_s']:.3f}",
        "",
    ])
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"psnr_db    = {metrics['psnr_db']:.2f}  (target >= {thr['psnr_db']} dB)",
        (
            "kwave_vs_analytical_pearson = "
            f"{r_kw_vs_ref:.6f}  "
            f"(target >= {analytical_thr['kwave_pearson_min']})"
        ),
        (
            "pkwav_vs_analytical_pearson = "
            f"{r_pkw_vs_ref:.6f}  "
            f"(target >= {analytical_thr['pykwavers_pearson_min']})"
        ),
        (
            "analytical_pearson_agreement_abs = "
            f"{abs(r_pkw_vs_ref - r_kw_vs_ref):.6e}  "
            f"(target < {analytical_thr['pearson_agreement_abs_max']:.6e})"
        ),
        f"peak_kwave_Pa      = {float(kw_amp.max()):.6e}",
        f"peak_pykwavers_Pa  = {float(pw_amp.max()):.6e}",
    ]
    save_text_report(METRICS_PATH, header, report_lines)
    print(f"  Saved metrics: {METRICS_PATH}")
    print(f"\n  Overall parity status: {status}")

    if status == "FAIL" and not args.allow_failure:
        raise SystemExit(
            "at_circular_piston_AS parity targets not met. "
            "Run with --allow-failure to collect diagnostics."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
