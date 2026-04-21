#!/usr/bin/env python3
"""
sd_focussed_detector_3D_compare.py
=====================================
Side-by-side comparison of k-wave-python vs pykwavers for the canonical
focused bowl detector example.  Two point pressure sources are simulated:
one at the geometric focus of a concave bowl sensor (on-axis) and one
laterally displaced (off-axis).  The bowl sensor spatially averages all
N bowl-surface pressure samples into a single time trace, producing a
much stronger response to the on-axis source — demonstrating the
directional selectivity of a focused single-element transducer.

Physical setup (matches k-wave-python `sd_focussed_detector_3D.py` exactly):
  Grid:     64×64×64,  dx = 100 mm / 64 = 1.5625 mm  (100 mm cubic domain)
  Medium:   homogeneous, c₀ = 1500 m/s, ρ = 1000 kg/m³, lossless
  Source:   CW sinusoid, f = 0.25 MHz, 1 Pa,
            band-limited by filter_time_series (shared between both engines)
  Sensor:   concave bowl, radius = 32 pts, aperture diameter = 33 pts
            bowl apex at [11, 32, 32] (1-indexed), focus at grid centre
  Source 1: on-axis point  → [sphere_offset + radius, Ny//2, Nz//2]
                            = [42, 31, 31]  (0-indexed)
  Source 2: off-axis point → [1 + sphere_offset + radius, Ny//2+5, Nz//2+5]
                            = [43, 37, 37]  (0-indexed)
  PML:      10 grid points, inside the 64³ grid (default k-wave pml_inside=True)

Bowl sensor data ordering
--------------------------
k-wave   : sensor_data["p"] → shape (Nt, n_bowl) or (n_bowl, Nt) depending on
           relative sizes. We detect by comparing shape[0] to kgrid.Nt.
           Sum over the spatial axis → (Nt,) spatially-summed time trace.
pykwavers: result.sensor_data → shape (n_bowl, Nt), sum axis=0 → (Nt,).

pykwavers source: pkw.Source.from_mask(mask, signal_1d, freq, mode="additive")
                  The filtered CW signal [Pa] is the same array used by k-wave.

Output:
  output/sd_focussed_detector_3D_compare.png    — 2×2 trace comparison figure
  output/sd_focussed_detector_3D_directivity.png — directivity overlay
  output/sd_focussed_detector_3D_metrics.txt     — parity metrics

Usage:
  python examples/sd_focussed_detector_3D_compare.py
  python examples/sd_focussed_detector_3D_compare.py --no-cache  # force re-run
  python examples/sd_focussed_detector_3D_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import sys
import time

# Ensure UTF-8 output on Windows (cp1252 terminals reject Unicode subscripts)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_bowl

# ---------------------------------------------------------------------------
# Grid / medium / source constants  (match k-wave-python EXACTLY)
# ---------------------------------------------------------------------------
N = 64                              # grid points per dimension
GRID_SIZE = Vector([N, N, N])
DX = 100e-3 / N                     # 1.5625e-3 m = 1.5625 mm
DX_VEC = Vector([DX, DX, DX])

C0   = 1500.0                       # sound speed [m/s]
RHO0 = 1000.0                       # density [kg/m³]

SOURCE_FREQ = 0.25e6                # CW carrier frequency [Hz]
SOURCE_MAG  = 1.0                   # source amplitude [Pa]

# Bowl sensor geometry (all positions 1-indexed, matching k-Wave convention)
SPHERE_OFFSET = 10                  # number of grid points from x=1 to bowl apex
BOWL_RADIUS   = N // 2             # 32
BOWL_DIAMETER = N // 2 + 1        # 33
BOWL_POS  = Vector([1 + SPHERE_OFFSET, N // 2, N // 2])   # [11, 32, 32]
FOCUS_POS = Vector([N // 2,           N // 2, N // 2])    # [32, 32, 32]

# Source positions (0-indexed NumPy), matching k-wave-python exactly:
#   source1[int(sphere_offset + radius), grid_size.y//2 - 1, grid_size.z//2 - 1]
#   source2[int(1+sphere_offset+radius), grid_size.y//2 + 5, grid_size.z//2 + 5]
SRC1_IX = int(SPHERE_OFFSET + BOWL_RADIUS)          # 42  (at focal plane)
SRC1_IY = N // 2 - 1                                # 31  (on-axis)
SRC1_IZ = N // 2 - 1                                # 31  (on-axis)

SRC2_IX = int(1 + SPHERE_OFFSET + BOWL_RADIUS)      # 43  (one step deeper)
SRC2_IY = N // 2 + 5                               # 37  (off-axis by 5 pts)
SRC2_IZ = N // 2 + 5                               # 37  (off-axis by 5 pts)

PML_SIZE = 10                       # default k-wave PML (pml_inside=True)

# ---------------------------------------------------------------------------
# Parity targets (CW trace comparison; looser than B-mode imaging)
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS = {
    "pearson_r":    0.93,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.25,
    "psnr_db":      18.0,
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
FIGURE_PATH      = DEFAULT_OUTPUT_DIR / "sd_focussed_detector_3D_compare.png"
DIR_FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "sd_focussed_detector_3D_directivity.png"
METRICS_PATH     = DEFAULT_OUTPUT_DIR / "sd_focussed_detector_3D_metrics.txt"

_KWAVE_CACHE  = {
    "src1": DEFAULT_OUTPUT_DIR / "sd_focussed_3D_kwave_src1.npz",
    "src2": DEFAULT_OUTPUT_DIR / "sd_focussed_3D_kwave_src2.npz",
}
_PKWAV_CACHE  = {
    "src1": DEFAULT_OUTPUT_DIR / "sd_focussed_3D_pykwavers_src1.npz",
    "src2": DEFAULT_OUTPUT_DIR / "sd_focussed_3D_pykwavers_src2.npz",
}


# ---------------------------------------------------------------------------
# Step 1 — Build shared configuration
# ---------------------------------------------------------------------------
def build_config() -> tuple:
    """
    Return (kgrid, medium, bowl_mask_bool, signal_1d, dt, Nt).

    The CW signal is band-limited once by k-wave's filter_time_series and
    shared verbatim with pykwavers, eliminating any signal-preprocessing
    discrepancy between the two legs.
    """
    kgrid  = kWaveGrid(GRID_SIZE, DX_VEC)
    medium = kWaveMedium(sound_speed=C0)
    kgrid.makeTime(medium.sound_speed)

    # Band-limited CW sinusoid (shared by both engines)
    raw_signal = SOURCE_MAG * np.sin(2.0 * np.pi * SOURCE_FREQ * kgrid.t_array)  # (1, Nt)
    filtered   = filter_time_series(kgrid, medium, raw_signal)
    signal_1d  = np.asarray(filtered, dtype=np.float64).flatten()  # (Nt,)

    # Concave bowl sensor mask (1-indexed positions, k-wave convention)
    bowl_mask      = make_bowl(GRID_SIZE, BOWL_POS, BOWL_RADIUS, BOWL_DIAMETER, FOCUS_POS)
    bowl_mask_bool = np.asarray(bowl_mask, dtype=bool)
    n_bowl         = int(bowl_mask_bool.sum())
    print(f"  Bowl sensor: {n_bowl} grid points  "
          f"(Nt={kgrid.Nt}, dt={kgrid.dt:.3e} s, t_end={kgrid.Nt*kgrid.dt*1e6:.1f} µs)")

    return kgrid, medium, bowl_mask_bool, signal_1d, float(kgrid.dt), int(kgrid.Nt)


# ---------------------------------------------------------------------------
# Step 2 — k-wave-python helper (single source position)
# ---------------------------------------------------------------------------
def _sum_kwave_bowl(p_data: np.ndarray, Nt: int) -> np.ndarray:
    """
    Spatially sum the bowl sensor data to a single time trace of shape (Nt,).

    k-wave returns sensor_data["p"] as either (Nt, n_bowl) or (n_bowl, Nt)
    depending on relative sizes.  We detect which by comparing axis 0 to Nt.
    """
    arr = np.asarray(p_data, dtype=np.float64)
    if arr.ndim == 1:
        return arr  # already (Nt,) — unexpected but safe
    if arr.shape[0] == Nt:
        return np.sum(arr, axis=1)   # (Nt, n_bowl) → sum over bowl → (Nt,)
    return np.sum(arr, axis=0)       # (n_bowl, Nt) → sum over bowl → (Nt,)


def run_kwave(src_tag: str,
              kgrid, medium,
              bowl_mask_bool: np.ndarray,
              signal_1d: np.ndarray,
              src_ix: int, src_iy: int, src_iz: int,
              use_gpu: bool = False) -> dict:
    """Run k-wave for one source position; return spatially-summed trace."""
    cache = _KWAVE_CACHE[src_tag]
    if cache.exists():
        print(f"  [k-wave {src_tag}] Loading from cache...")
        d = np.load(cache)
        return {"trace": d["trace"], "dt": float(d["dt"]), "runtime_s": float(d["runtime_s"])}

    # Source: point pressure at (src_ix, src_iy, src_iz)
    src_mask = np.zeros((N, N, N), dtype=np.float64)
    src_mask[src_ix, src_iy, src_iz] = 1.0

    source       = kSource()
    source.p_mask = src_mask
    source.p      = signal_1d.reshape(1, -1)   # (1, Nt) — k-wave expects 2-D

    sensor = kSensor(bowl_mask_bool.astype(np.int32))

    sim_opts  = SimulationOptions(pml_size=PML_SIZE, data_cast="single", save_to_disk=True)
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=use_gpu)

    print(f"  [k-wave {src_tag}] Running...")
    t0 = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium, kgrid=kgrid,
        source=source, sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [k-wave {src_tag}] Done in {elapsed:.1f} s")

    trace  = _sum_kwave_bowl(sensor_data["p"], kgrid.Nt)
    result = {"trace": trace, "dt": float(kgrid.dt), "runtime_s": elapsed}
    np.savez(cache, **result)
    return result


# ---------------------------------------------------------------------------
# Step 3 — pykwavers simulation helper (single source position)
# ---------------------------------------------------------------------------
def run_pykwavers(src_tag: str,
                  bowl_mask_bool: np.ndarray,
                  signal_1d: np.ndarray,
                  src_ix: int, src_iy: int, src_iz: int,
                  dt: float, Nt: int) -> dict:
    """
    Run pykwavers CPU PSTD for one source position.

    Source injection: pkw.Source.from_mask(mask, signal, freq, mode="additive")
    applies signal_1d [Pa] additively at the single mask point — equivalent to
    k-wave's source.p_mask + source.p with mode "additive".

    Sensor: pkw.Sensor.from_mask(bowl_mask_bool)
    result.sensor_data shape: (n_bowl, Nt) — sum axis=0 → (Nt,) trace.
    """
    cache = _PKWAV_CACHE[src_tag]
    if cache.exists():
        print(f"  [pykwavers {src_tag}] Loading from cache...")
        d = np.load(cache)
        return {"trace": d["trace"], "runtime_s": float(d["runtime_s"])}

    # Source mask (float64, single non-zero point)
    src_mask = np.zeros((N, N, N), dtype=np.float64)
    src_mask[src_ix, src_iy, src_iz] = 1.0

    grid   = pkw.Grid(N, N, N, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    source = pkw.Source.from_mask(src_mask, signal_1d, SOURCE_FREQ, mode="additive")
    sensor = pkw.Sensor.from_mask(bowl_mask_bool)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print(f"  [pykwavers {src_tag}] Running...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=Nt, dt=dt)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers {src_tag}] Done in {elapsed:.1f} s")

    # result.sensor_data: (n_bowl, Nt) → sum over bowl → (Nt,)
    sd    = np.asarray(result.sensor_data, dtype=np.float64)
    trace = np.sum(sd, axis=0)

    output = {"trace": trace, "runtime_s": elapsed}
    np.savez(cache, **output)
    return output


# ---------------------------------------------------------------------------
# Step 4 — Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw1: dict, kw2: dict, pkw1: dict, pkw2: dict, dt: float) -> None:
    """
    2×2 figure:
      Rows: Source 1 (on-axis) / Source 2 (off-axis)
      Cols: Full time trace / Zoom on last 5 periods (steady state)
    """
    Nt         = len(kw1["trace"])
    t_us       = np.arange(Nt) * dt * 1e6
    period_us  = 1e6 / SOURCE_FREQ              # 4 µs at 0.25 MHz
    n_zoom     = max(1, int(5 * period_us / (dt * 1e6)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    rows = [
        (kw1, pkw1, "Source 1 — on-axis (at geometric focus)"),
        (kw2, pkw2, "Source 2 — off-axis (Δy = Δz = 5 grid pts)"),
    ]
    for row, (kw, pkw_res, label) in enumerate(rows):
        kw_tr  = kw["trace"]
        pkw_tr = pkw_res["trace"]

        # Full trace
        ax = axes[row, 0]
        ax.plot(t_us, kw_tr,  "k-",  linewidth=1.5, alpha=0.85, label="k-wave-python")
        ax.plot(t_us, pkw_tr, "r--", linewidth=1.2, alpha=0.85, label="pykwavers")
        ax.set_xlabel("Time [µs]")
        ax.set_ylabel("Summed bowl pressure [Pa]")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Zoom: last 5 periods (steady state)
        ax2 = axes[row, 1]
        ax2.plot(t_us[-n_zoom:], kw_tr[-n_zoom:],  "k-",  linewidth=1.5, alpha=0.85, label="k-wave-python")
        ax2.plot(t_us[-n_zoom:], pkw_tr[-n_zoom:], "r--", linewidth=1.2, alpha=0.85, label="pykwavers")
        ax2.set_xlabel("Time [µs]")
        ax2.set_ylabel("Summed bowl pressure [Pa]")
        ax2.set_title(f"{label}\nsteady-state zoom (last 5 periods)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "sd_focussed_detector_3D: k-wave-python vs pykwavers\n"
        f"Concave bowl sensor  ·  f={SOURCE_FREQ * 1e-3:.0f} kHz  ·  grid {N}³"
        f"  ·  dx={DX * 1e3:.4f} mm",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


def plot_directivity(kw1: dict, kw2: dict, pkw1: dict, pkw2: dict, dt: float) -> None:
    """
    Single panel showing all four traces on one axis to illustrate
    the directivity contrast between on-axis and off-axis sources.
    """
    t_us = np.arange(len(kw1["trace"])) * dt * 1e6

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_us, kw1["trace"],  "b-",  linewidth=1.8, label="k-wave: on-axis",  alpha=0.9)
    ax.plot(t_us, kw2["trace"],  "b--", linewidth=1.2, label="k-wave: off-axis", alpha=0.65)
    ax.plot(t_us, pkw1["trace"], "r-",  linewidth=1.4, label="pykwavers: on-axis",  alpha=0.85)
    ax.plot(t_us, pkw2["trace"], "r--", linewidth=0.9, label="pykwavers: off-axis", alpha=0.55)
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Summed bowl pressure [Pa]")
    ax.set_title(
        "Focused bowl detector directivity\n"
        "on-axis source (at focus) vs off-axis source (Δy=Δz=5 pts)"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(DIR_FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {DIR_FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for sd_focussed_detector_3D."
    )
    parser.add_argument("--gpu",           action="store_true",
                        help="Run k-wave-python with GPU execution binary.")
    parser.add_argument("--no-cache",      action="store_true",
                        help="Delete cached results and force a fresh run.")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail (diagnostics).")
    args = parser.parse_args()

    if args.no_cache:
        for cache_dict in [_KWAVE_CACHE, _PKWAV_CACHE]:
            for p in cache_dict.values():
                if p.exists():
                    p.unlink()
                    print(f"  Removed cache: {p}")

    print("=" * 62)
    print("sd_focussed_detector_3D: k-wave-python vs pykwavers")
    print(f"  Grid   : {N}³   dx = {DX * 1e3:.4f} mm   (100 mm domain)")
    print(f"  Medium : c₀ = {C0} m/s  lossless  (density = {RHO0} kg/m³)")
    print(f"  Source : CW {SOURCE_FREQ * 1e-3:.0f} kHz, {SOURCE_MAG:.0f} Pa (band-limited)")
    print(f"  PML    : {PML_SIZE} pts (inside)")
    print(f"  Src 1  : on-axis   [{SRC1_IX}, {SRC1_IY}, {SRC1_IZ}]  (at focus)")
    print(f"  Src 2  : off-axis  [{SRC2_IX}, {SRC2_IY}, {SRC2_IZ}]  (Δy = Δz = 5 pts)")
    print(f"  Bowl   : radius={BOWL_RADIUS} pts, diameter={BOWL_DIAMETER} pts")
    print("=" * 62)

    # --- Shared configuration ---
    print("\n[1/5] Building configuration...")
    kgrid, medium, bowl_mask_bool, signal_1d, dt, Nt = build_config()

    # --- k-wave: source 1 ---
    print("\n[2/5] k-wave-python — source 1 (on-axis)...")
    kw1 = run_kwave("src1", kgrid, medium, bowl_mask_bool, signal_1d,
                    SRC1_IX, SRC1_IY, SRC1_IZ, use_gpu=args.gpu)
    print(f"  peak = {np.abs(kw1['trace']).max():.4f} Pa   rms = {np.sqrt(np.mean(kw1['trace']**2)):.4f} Pa")

    # --- k-wave: source 2 ---
    print("\n[3/5] k-wave-python — source 2 (off-axis)...")
    kw2 = run_kwave("src2", kgrid, medium, bowl_mask_bool, signal_1d,
                    SRC2_IX, SRC2_IY, SRC2_IZ, use_gpu=args.gpu)
    print(f"  peak = {np.abs(kw2['trace']).max():.4f} Pa   rms = {np.sqrt(np.mean(kw2['trace']**2)):.4f} Pa")

    # --- pykwavers: source 1 ---
    print("\n[4/5] pykwavers — source 1 (on-axis)...")
    pkw1 = run_pykwavers("src1", bowl_mask_bool, signal_1d,
                          SRC1_IX, SRC1_IY, SRC1_IZ, dt, Nt)
    print(f"  peak = {np.abs(pkw1['trace']).max():.4f} Pa   rms = {np.sqrt(np.mean(pkw1['trace']**2)):.4f} Pa")

    # --- pykwavers: source 2 ---
    print("\n[5/5] pykwavers — source 2 (off-axis)...")
    pkw2 = run_pykwavers("src2", bowl_mask_bool, signal_1d,
                          SRC2_IX, SRC2_IY, SRC2_IZ, dt, Nt)
    print(f"  peak = {np.abs(pkw2['trace']).max():.4f} Pa   rms = {np.sqrt(np.mean(pkw2['trace']**2)):.4f} Pa")

    # --- Parity metrics ---
    print("\n--- Parity evaluation ---")
    all_pass       = True
    report_sections: list[str] = []
    thr = PARITY_THRESHOLDS

    for label, kw, pkw_res in [
        ("src1_on_axis",   kw1, pkw1),
        ("src2_off_axis",  kw2, pkw2),
    ]:
        metrics = compute_image_metrics(kw["trace"], pkw_res["trace"])
        checks  = {
            "pearson_r":  metrics["pearson_r"]  >= thr["pearson_r"],
            "rms_ratio":  thr["rms_ratio_min"]  <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
            "psnr_db":    metrics["psnr_db"]    >= thr["psnr_db"],
        }
        status   = "PASS" if all(checks.values()) else "FAIL"
        all_pass = all_pass and (status == "PASS")

        kw_peak  = float(np.abs(kw["trace"]).max())
        pkw_peak = float(np.abs(pkw_res["trace"]).max())

        print(f"  [{label}]  {status}")
        print(f"    Pearson r   = {metrics['pearson_r']:.6f}  "
              f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
        print(f"    RMS ratio   = {metrics['rms_ratio']:.6f}  "
              f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
              f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
        print(f"    PSNR        = {metrics['psnr_db']:.2f} dB  "
              f"(target >= {thr['psnr_db']} dB)  {'OK' if checks['psnr_db'] else 'FAIL'}")
        print(f"    Peak kwave  = {kw_peak:.4f} Pa")
        print(f"    Peak pkwav  = {pkw_peak:.4f} Pa  (ratio={pkw_peak/(kw_peak+1e-30):.4f})")

        report_sections.extend([
            f"{label}: {status}",
            f"  pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
            f"  rms_ratio  = {metrics['rms_ratio']:.6f}  "
            f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
            f"  psnr_db    = {metrics['psnr_db']:.2f}  (target >= {thr['psnr_db']} dB)",
            f"  peak_kwave_Pa     = {kw_peak:.6e}",
            f"  peak_pykwavers_Pa = {pkw_peak:.6e}",
            f"  peak_ratio        = {pkw_peak / (kw_peak + 1e-30):.6f}",
            "",
        ])

    # Directivity test: on-axis peak > off-axis peak for both engines
    kw_dir_ratio  = float(np.abs(kw1["trace"]).max())  / (float(np.abs(kw2["trace"]).max())  + 1e-30)
    pkw_dir_ratio = float(np.abs(pkw1["trace"]).max()) / (float(np.abs(pkw2["trace"]).max()) + 1e-30)
    dir_pass = kw_dir_ratio > 1.0 and pkw_dir_ratio > 1.0

    print(f"\n  Directivity (on-axis peak / off-axis peak):")
    print(f"    k-wave-python : {kw_dir_ratio:.3f}  {'OK' if kw_dir_ratio > 1 else 'FAIL'}")
    print(f"    pykwavers     : {pkw_dir_ratio:.3f}  {'OK' if pkw_dir_ratio > 1 else 'FAIL'}")
    all_pass = all_pass and dir_pass

    report_sections.extend([
        "directivity_test:",
        f"  kwave_on_off_ratio    = {kw_dir_ratio:.4f}",
        f"  pykwavers_on_off_ratio = {pkw_dir_ratio:.4f}",
        f"  status = {'PASS' if dir_pass else 'FAIL'}",
        "",
    ])

    overall = "PASS" if all_pass else "FAIL"

    # --- Figures ---
    plot_comparison(kw1, kw2, pkw1, pkw2, dt)
    plot_directivity(kw1, kw2, pkw1, pkw2, dt)

    # --- Text report ---
    header = "\n".join([
        "sd_focussed_detector_3D parity metrics",
        f"parity_status: {overall}",
        f"grid: {N}x{N}x{N}   dx={DX:.6e} m",
        f"source: CW {SOURCE_FREQ*1e-3:.0f} kHz  {SOURCE_MAG:.0f} Pa  band-limited",
        f"bowl: apex=[{int(BOWL_POS.x)},{int(BOWL_POS.y)},{int(BOWL_POS.z)}] 1-idx, "
        f"radius={BOWL_RADIUS}, diameter={BOWL_DIAMETER}",
        f"dt={dt:.6e} s   Nt={Nt}",
        f"kwave_src1_runtime_s: {kw1['runtime_s']:.3f}",
        f"kwave_src2_runtime_s: {kw2['runtime_s']:.3f}",
        f"pykwavers_src1_runtime_s: {pkw1['runtime_s']:.3f}",
        f"pykwavers_src2_runtime_s: {pkw2['runtime_s']:.3f}",
        "",
    ])
    save_text_report(METRICS_PATH, header, report_sections)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {overall}")

    if not all_pass and not args.allow_failure:
        raise SystemExit(
            "sd_focussed_detector_3D parity targets not met. "
            "Run with --allow-failure to collect diagnostics."
        )

    print("\nDone.")
    print(f"  Figures: {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
