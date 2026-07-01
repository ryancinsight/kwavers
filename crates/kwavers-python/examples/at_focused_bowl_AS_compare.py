#!/usr/bin/env python3
"""
at_focused_bowl_AS_compare.py
=============================
Side-by-side comparison of k-wave-python (`kspaceFirstOrderASC`) vs pykwavers
PSTD axisymmetric solver for the focused bowl transducer example.

Physical setup (matches k-wave-python `at_focused_bowl_AS.py`):
    Grid:      (Nx, Ny) AS — Nx = axial, Ny = radial (Nr)
    Medium:    homogeneous, c0 = 1500 m/s, rho = 1000 kg/m³, lossless
    Source:    arc element (WSWA-FFT BLI), roc = 30 mm, dia = 30 mm
               driven at 1 MHz CW, 1 MPa amplitude
    Sensor:    full 2-D mask (all points past source)
    PML:       outside (pml_inside=False)
    t_end:     40 µs (enough to reach steady state at 1 MHz)

Validation:
    On-axis amplitude profile (steady-state) compared between k-wave reference
    and pykwavers AS solver using a source mask taken directly from k-wave-python's
    BLI engine (source.p_mask and source.p).

    The pykwavers source is constructed via Source.from_mask(mask_3d_float, signal)
    where mask_3d has shape (Nx, 1, Ny) and signal rows match k-wave's distributed
    signal order (Fortran-column active-cell enumeration).

Usage:
    python examples/at_focused_bowl_AS_compare.py
    python examples/at_focused_bowl_AS_compare.py --no-cache
    python examples/at_focused_bowl_AS_compare.py --allow-failure
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
from kwave.utils.mapgen import focused_bowl_oneil
from kwave.utils.math import round_even
from kwave.utils.signals import create_cw_signals

# ---------------------------------------------------------------------------
# Physical parameters (from at_focused_bowl_AS.py)
# ---------------------------------------------------------------------------
C0            = 1500.0      # [m/s]
RHO0          = 1000.0      # [kg/m³]
SOURCE_F0     = 1.0e6       # [Hz]
SOURCE_ROC    = 30e-3       # bowl radius of curvature [m]
SOURCE_DIA    = 30e-3       # bowl aperture diameter [m]
SOURCE_AMP    = np.array([1.0e6])  # [Pa]
SOURCE_PHASE  = np.array([0.0])    # [rad]
AXIAL_SIZE    = 50e-3       # [m]
LATERAL_SIZE  = 45e-3       # [m]
PPW           = 3           # points per wavelength
T_END         = 40e-6       # [s]
RECORD_PERIODS = 1
CFL           = 0.05
SOURCE_X_OFFSET = 20        # grid points
BLI_TOL       = 0.01
UPSAMPLE_RATE = 10

# Grid spacing
DX = C0 / (PPW * SOURCE_F0)     # [m]

# Grid sizes (k-wave convention: 2D Nx×Ny, axial×radial)
NX = round_even(AXIAL_SIZE / DX) + SOURCE_X_OFFSET
NY = round_even(LATERAL_SIZE / DX)

# Time discretisation
PPP = round(PPW / CFL)
DT  = 1.0 / (PPP * SOURCE_F0)
NT  = round(T_END / DT)

# For pykwavers: grid is (NX, 1, NY) in AS mode
# Source offset in k-wave: arc_pos[0] = kgrid.x_vec[0] + SOURCE_X_OFFSET * DX
# In pykwavers world (x from 0): source_x = SOURCE_X_OFFSET * DX

# Sensor excludes the source rows (source_x_offset + 1 : )
SENSOR_START  = SOURCE_X_OFFSET + 1   # first axial index with sensor
N_SENSOR_AX   = NX - SENSOR_START     # number of axial sensor points

# Parity thresholds (on-axis amplitude vs analytical O'Neil solution)
PARITY_THRESHOLDS = {
    "pearson_r":     0.90,
    "rms_ratio_min": 0.75,
    "rms_ratio_max": 1.35,
    "psnr_db":       12.0,
}
ANALYTICAL_REFERENCE_THRESHOLDS = {
    "pearson_agreement_abs_max": 1e-5,
}

# Paths
FIGURE_PATH       = DEFAULT_OUTPUT_DIR / "at_focused_bowl_AS_compare.png"
METRICS_PATH      = DEFAULT_OUTPUT_DIR / "at_focused_bowl_AS_metrics.txt"
KWAVE_CACHE       = DEFAULT_OUTPUT_DIR / "at_focused_bowl_AS_kwave_cache.npz"
KWAVE_RAW_CACHE   = DEFAULT_OUTPUT_DIR / "at_focused_bowl_AS_kwave_raw.npz"
PKWAV_CACHE       = DEFAULT_OUTPUT_DIR / "at_focused_bowl_AS_pykwavers_cache.npz"
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
    arc_pos   = [kgrid.x_vec[0].item() + SOURCE_X_OFFSET * kgrid.dx, 0]
    focus_pos = [kgrid.x_vec[-1].item(), 0]
    karray.add_arc_element(arc_pos, SOURCE_ROC, SOURCE_DIA, focus_pos)

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)       # (NX, NY) bool
    source.p      = karray.get_distributed_source_signal(kgrid, source_sig)  # (n_active, NT)
    return source, karray


# ---------------------------------------------------------------------------
# Run k-wave-python (reference)
# ---------------------------------------------------------------------------
def run_kwave() -> dict:
    if KWAVE_CACHE.exists():
        print("  [k-wave] Loading from cache...")
        d = np.load(KWAVE_CACHE)
        return {"amp_on_axis": d["amp_on_axis"], "runtime_s": float(d["runtime_s"])}

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
        sensor.mask = np.zeros((NX, NY), dtype=bool)
        sensor.mask[SENSOR_START:, :] = True
        sensor.record = ["p"]
        sensor.record_start_index = kgrid.Nt - RECORD_PERIODS * PPP + 1

        sim_opts = SimulationOptions(
            simulation_type=SimulationType.AXISYMMETRIC,
            pml_inside=False,
            save_to_disk=True,
            save_to_disk_exit=False,
        )
        exec_opts = SimulationExecutionOptions(is_gpu_simulation=False, delete_data=False)

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

    # Reshape: sensor spans (SENSOR_START:NX, 0:NY) in k-wave Fortran column-major order
    amp_2d = np.reshape(amp, (N_SENSOR_AX, NY), order="F")
    amp_on_axis = amp_2d[:, 0]

    result = {"amp_on_axis": amp_on_axis, "runtime_s": elapsed}
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
            return {"amp_on_axis": d["amp_on_axis"], "runtime_s": float(d["runtime_s"])}

    # Use k-wave-python to generate the BLI source mask and distributed signal.
    # This ensures pykwavers uses the same source weights as the k-wave reference.
    kgrid  = _build_kwave_grid()
    source, _ = _build_kwave_source(kgrid)

    # source.p_mask: (NX, NY) bool array  — binary mask in axial×radial (Fortran layout)
    # source.p:      (n_active, NT) float — distributed signal per active cell (Fortran order)
    p_mask_2d = np.asarray(source.p_mask, dtype=np.float64)   # (NX, NY)
    p_sig_2d  = np.asarray(source.p,      dtype=np.float64)   # (n_active, NT)

    # pykwavers grid is (NX, 1, NY): axial × trivial × radial
    grid   = pkw.Grid(NX, 1, NY, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    # Convert 2D mask to 3D (NX, 1, NY) in C order
    p_mask_3d = p_mask_2d[:, np.newaxis, :]   # (NX, 1, NY)

    # Source.from_mask accepts: float 3D mask (weights), signal (n_active or 1, NT), freq
    # The active cells in p_mask_3d must match p_sig_2d rows.
    # k-wave enumerates active cells in Fortran column-major order over the 2D mask.
    # pykwavers enumerates active cells in C row-major order over the 3D mask.
    # Since ny=1, the column order over (NX, 1, NY) matches Fortran order over (NX, NY)
    # when the column axis is NY (last index) — verified: both scan fastest over radial.
    pkw_source = pkw.Source.from_mask(p_mask_3d, p_sig_2d, SOURCE_F0, mode="additive")

    # Sensor: full 2D slice past source, stored as (NX, 1, NY) mask
    sensor_mask = np.zeros((NX, 1, NY), dtype=bool)
    sensor_mask[SENSOR_START:, 0, :] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, pkw_source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_axisymmetric(True)
    sim.set_pml_inside(False)

    print("  [pykwavers] Running PSTD AS...")
    t0 = time.perf_counter()
    result_run = sim.run(time_steps=NT, dt=DT)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    # result_run.sensor_data: (n_sensor, NT)
    p = np.asarray(result_run.sensor_data, dtype=np.float64)
    # Take only the last RECORD_PERIODS * PPP timesteps (steady-state)
    n_record = RECORD_PERIODS * PPP
    p_ss = p[:, -n_record:]

    amp, _, _ = extract_amp_phase(p_ss, 1.0 / DT, SOURCE_F0, dim=1, fft_padding=1, window="Rectangular")
    amp = np.asarray(amp, dtype=np.float64).flatten()

    # Sensor recorder emits Fortran-order rows to match MATLAB / k-Wave `find`.
    # Reshape in the same order before selecting the on-axis radial slice.
    n_radial = NY
    amp_2d = amp.reshape(N_SENSOR_AX, n_radial, order="F")
    amp_on_axis = amp_2d[:, 0]

    output = {"amp_on_axis": amp_on_axis, "runtime_s": elapsed}
    np.savez(PKWAV_CACHE, cache_version=PKWAV_CACHE_VERSION, **output)
    return output


# ---------------------------------------------------------------------------
# Analytical O'Neil reference
# ---------------------------------------------------------------------------
def analytical_oneil(x_vec_m: np.ndarray) -> np.ndarray:
    """Return O'Neil on-axis pressure amplitude [Pa] at axial positions x_vec_m."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ax, _, _ = focused_bowl_oneil(
            SOURCE_ROC, SOURCE_DIA, SOURCE_AMP[0] / (C0 * RHO0),
            SOURCE_F0, C0, RHO0, axial_positions=x_vec_m,
        )
    return np.asarray(p_ax, dtype=np.float64)


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
    print("at_focused_bowl_AS: k-wave-python vs pykwavers")
    print(f"  Grid     : {NX}×{NY} (axial×radial)  dx={DX*1e3:.3f} mm")
    print(f"  Medium   : c0={C0} m/s  rho={RHO0} kg/m³  lossless")
    print(f"  Source   : arc (BLI)  roc={SOURCE_ROC*1e3:.0f} mm  dia={SOURCE_DIA*1e3:.0f} mm")
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

    # Also compare vs O'Neil analytical solution.
    # Axial positions must be measured from the bowl face (source grid index =
    # SOURCE_X_OFFSET), not from the grid origin.  Using absolute grid-index
    # positions shifts the O'Neil peak by SOURCE_X_OFFSET cells relative to the
    # simulated focus, driving the Pearson correlation near zero.
    x_from_source = np.arange(SENSOR_START - SOURCE_X_OFFSET, NX - SOURCE_X_OFFSET) * DX
    p_ref = analytical_oneil(x_from_source)
    # focused_bowl_oneil has a formal singularity at x == roc (geometric focus);
    # mask those points before computing Pearson so NaN does not propagate.
    finite_mask = np.isfinite(p_ref)

    r_vs_ref, _ = pearsonr(p_ref[finite_mask], kw_amp[finite_mask])
    r_vs_pkw, _ = pearsonr(p_ref[finite_mask], pw_amp[finite_mask])

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
    print(f"\n--- vs O'Neil analytical ---")
    analytical_thr = ANALYTICAL_REFERENCE_THRESHOLDS
    print(f"  k-wave vs analytical   pearson r = {r_vs_ref:.4f}")
    print(
        "  pykwavers vs analytical pearson r = "
        f"{r_vs_pkw:.4f}  "
        f"(|Δr| < {analytical_thr['pearson_agreement_abs_max']:.6e})"
    )
    print(f"\n  k-wave runtime     : {kw['runtime_s']:.1f} s")
    print(f"  pykwavers runtime  : {pw['runtime_s']:.1f} s")

    # Plot — x-axis is distance from the bowl face (source-relative)
    x_mm = x_from_source * 1e3
    x_ref_mm = np.linspace(0, (NX - SOURCE_X_OFFSET) * DX * 1e3, 10000)
    p_ref_full = analytical_oneil(x_ref_mm * 1e-3)
    finite_ref_full = np.isfinite(p_ref_full)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x_ref_mm[finite_ref_full],
        p_ref_full[finite_ref_full] * 1e-6,
        "k-",
        lw=1.4,
        label="O'Neil analytical",
    )
    ax.plot(x_mm, kw_amp * 1e-6, "b-", lw=1.6, alpha=0.8, label="k-wave (kspaceFirstOrderASC)")
    ax.plot(x_mm, pw_amp * 1e-6, "r--", lw=1.4, alpha=0.85, label="pykwavers (WSWA-FFT AS)")
    ax.axvline(SOURCE_ROC * 1e3, color="gray", ls=":", lw=1.0, label=f"focus ({SOURCE_ROC*1e3:.0f} mm)")
    ax.set_xlabel("Axial distance from source [mm]")
    ax.set_ylabel("Steady-state pressure amplitude [MPa]")
    ax.set_title(
        f"at_focused_bowl_AS: k-wave vs pykwavers  |  {status}\n"
        f"roc={SOURCE_ROC*1e3:.0f} mm  dia={SOURCE_DIA*1e3:.0f} mm  "
        f"f0={SOURCE_F0*1e-6:.1f} MHz  {NX}×{NY} grid  dx={DX*1e6:.0f} µm"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved figure : {FIGURE_PATH}")

    header = "\n".join([
        "at_focused_bowl_AS parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY} (axial x radial)  dx={DX:.6e} m",
        f"source: arc  roc={SOURCE_ROC} m  dia={SOURCE_DIA} m  amp={SOURCE_AMP[0]} Pa  f0={SOURCE_F0} Hz",
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
        f"kwave_vs_analytical_pearson = {r_vs_ref:.6f}",
        f"pkwav_vs_analytical_pearson = {r_vs_pkw:.6f}",
        (
            "analytical_pearson_agreement_abs = "
            f"{abs(r_vs_ref - r_vs_pkw):.6e}  "
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
            "at_focused_bowl_AS parity targets not met. "
            "Run with --allow-failure to collect diagnostics."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
