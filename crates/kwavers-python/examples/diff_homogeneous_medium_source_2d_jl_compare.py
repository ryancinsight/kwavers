#!/usr/bin/env python3
"""
diff_homogeneous_medium_source_2d_jl_compare.py
================================================
Parity comparison of KWave.jl ``kwave_diffusion`` (2D) against pykwavers
``ThermalSimulation`` for the canonical MATLAB k-Wave example:

    example_diff_homogeneous_medium_source.m
    http://www.k-wave.org/documentation/example_diff_homogeneous_medium_source.php

Physical setup (soft tissue, matched to k-Wave MATLAB example parameters):
    Grid:        128 × 128 active points, dx = dy = 0.5 mm
    Domain:      64 mm × 64 mm
    Medium:      κ=0.52 W/(m·K), ρ=1040 kg/m³, cp=3650 J/(kg·K)
    Perfusion:   w_b = 0.009 1/s (brain), ρ_b=1050, c_b=3617
    Source:      Q = Q0 · exp(−r²/(2σ²)),  Q0=1e6 W/m³, σ=3 mm  (Gaussian disk)
    Time:        dt=0.05 s, Nt=600 (30 s total)
    Initial T:   37 °C (body temperature)

Solver comparison
-----------------
KWave.jl kwave_diffusion uses a pseudospectral (FFT-based) Laplacian with
forward-Euler time integration and periodic boundary conditions.

pykwavers ThermalSimulation uses a finite-difference Laplacian (interior
only; boundary voxels have zero Laplacian) with forward-Euler time steps.

When the Gaussian source has σ ≪ domain/2 (here σ=2 mm, domain=32 mm →
σ/half-domain ≈ 0.125), both BCs produce identical results in the interior:
the temperature distribution decays to < 10⁻¹⁰ × peak before reaching any
boundary, so periodic vs zero-flux differences are below numerical noise.
The pykwavers slab (nx=128, ny=128, nz=1) is identical to the 2D KWave.jl
grid because the z-Laplacian is identically zero (no z-variation).

Parity criteria (matched-engine comparison, same physics and numerics):
    pearson_r  >= 0.9999
    rms_ratio  in [0.999, 1.001]
    psnr_db    >= 60 dB

Outputs
-------
    output/diff_homogeneous_medium_source_2d_jl_compare.png
    output/diff_homogeneous_medium_source_2d_jl_metrics.txt
    output/diff_homogeneous_medium_source_2d_jl_meta.json
    output/diff_homogeneous_medium_source_2d_jl_kwave_cache.npy  (cached)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()
import pykwavers as pkw

REPO_ROOT = HERE.parents[2]
JULIA_PROJECT = REPO_ROOT / "external" / "k-wave-julia" / "KWave.jl"
JULIA_DRIVER = HERE / "run_kwave_julia_diff_homogeneous_medium_source_2d.jl"

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH  = OUTPUT_DIR / "diff_homogeneous_medium_source_2d_jl_compare.png"
METRICS_PATH = OUTPUT_DIR / "diff_homogeneous_medium_source_2d_jl_metrics.txt"
META_PATH    = OUTPUT_DIR / "diff_homogeneous_medium_source_2d_jl_meta.json"
CACHE_PATH   = OUTPUT_DIR / "diff_homogeneous_medium_source_2d_jl_kwave_cache.npy"
CACHE_META   = OUTPUT_DIR / "diff_homogeneous_medium_source_2d_jl_kwave_cache_meta.json"

# ---------------------------------------------------------------------------
# Physical parameters (match k-Wave MATLAB example_diff_homogeneous_medium_source.m)
# ---------------------------------------------------------------------------
NX, NY     = 128, 128
DX = DY    = 0.5e-3             # 0.5 mm → 64 mm domain

K_TH       = 0.52               # thermal conductivity [W/(m·K)]
RHO        = 1040.0             # density [kg/m³]
CP         = 3650.0             # specific heat [J/(kg·K)]
WB_PER_S   = 0.009              # perfusion [1/s]
RHO_B      = 1050.0             # blood density [kg/m³]
CP_B       = 3617.0             # blood specific heat [J/(kg·K)]
TA         = 37.0               # arterial/initial temperature [°C]

# KWave.jl perfusion_rate is in [kg/(m³·s)]: perfusion_rate_jl = wb[1/s] * rho_b
WB_JL      = WB_PER_S * RHO_B  # 0.009 * 1050 = 9.45 kg/(m³·s)

Q0         = 1e6               # Gaussian source peak [W/m³]
Q_SIGMA    = 3.0e-3            # Gaussian σ [m] = 3 mm (6 cells; σ/half-domain = 0.094)

# Stability: KWave.jl uses forward-Euler + spectral Laplacian (periodic BC).
# dt_max = ρ·cp·dx² / (2π²·κ) = 3796000·2.5e-7 / (2·9.87·0.52) ≈ 0.092 s
# pykwavers uses FD Laplacian (interior only); dt_max ≈ 0.114 s.
# DT=0.05 s gives safety margin ~0.54 for KWave.jl.
DT         = 0.05              # time step [s]
NT         = 600               # 30 s total

PARITY_THRESHOLDS = {
    "pearson_r":      0.9999,
    "rms_ratio_min":  0.999,
    "rms_ratio_max":  1.001,
    "psnr_db":        60.0,
}


# ---------------------------------------------------------------------------
# Step 1 — Run KWave.jl (with caching)
# ---------------------------------------------------------------------------
def run_kwave_julia(force: bool = False) -> np.ndarray:
    """Run KWave.jl 2D kwave_diffusion and return final T field (NX, NY)."""
    if not force and CACHE_PATH.exists() and CACHE_META.exists():
        print("  [KWave.jl] Loading from cache...")
        return np.load(str(CACHE_PATH))

    tmp_csv  = str(OUTPUT_DIR / "_jl2d_T_final.csv")
    tmp_meta = str(OUTPUT_DIR / "_jl2d_meta.json")

    cmd = [
        "julia",
        f"--project={JULIA_PROJECT}",
        str(JULIA_DRIVER),
        "--nx",          str(NX),
        "--ny",          str(NY),
        "--dx",          str(DX),
        "--dy",          str(DY),
        "--nt",          str(NT),
        "--dt",          str(DT),
        "--thermal-conductivity",   str(K_TH),
        "--density",                str(RHO),
        "--specific-heat",          str(CP),
        "--perfusion-rate",         str(WB_JL),
        "--blood-temperature",      str(TA),
        "--blood-specific-heat",    str(CP_B),
        "--initial-temperature",    str(TA),
        "--q-peak",     str(Q0),
        "--q-sigma",    str(Q_SIGMA),
        "--output-csv",  tmp_csv,
        "--output-meta", tmp_meta,
    ]

    print("  [KWave.jl] Running 2D diffusion simulation...")
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = "4"
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("JULIA STDERR:\n", result.stderr[-3000:])
        raise RuntimeError(f"Julia exited with code {result.returncode}")

    print(f"  [KWave.jl] Done in {elapsed:.1f} s")
    print("  ", result.stdout.strip().split("\n")[-1])

    # Parse row-major CSV: row=ix, col=iy (skipping comment header)
    T_jl = np.zeros((NX, NY), dtype=np.float64)
    with open(tmp_csv) as f:
        ix = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.split(",")]
            T_jl[ix, :] = vals
            ix += 1
    assert ix == NX, f"Expected {NX} rows, got {ix}"

    np.save(str(CACHE_PATH), T_jl)
    with open(CACHE_META, "w") as f:
        json.dump(json.load(open(tmp_meta)), f, indent=2)

    os.remove(tmp_csv)
    os.remove(tmp_meta)
    return T_jl


# ---------------------------------------------------------------------------
# Step 2 — Run pykwavers ThermalSimulation (2D slab: nz=1)
# ---------------------------------------------------------------------------
def run_pykwavers() -> np.ndarray:
    """Run pykwavers ThermalSimulation on a 2D slab (nz=1) and return T field (NX, NY)."""
    print("  [pykwavers] Building 2D heat source...")

    # Build 3D Gaussian source on (NX, NY, 1) slab.
    # Julia driver uses cx = nx ÷ 2 = 64 (1-indexed) = cell 63 (0-indexed).
    # Match this convention: cx_py = NX // 2 - 1 = 63 for NX=128.
    cx, cy = NX // 2 - 1, NY // 2 - 1
    ix_arr = np.arange(NX, dtype=np.float64)
    iy_arr = np.arange(NY, dtype=np.float64)
    IX, IY = np.meshgrid(ix_arr, iy_arr, indexing="ij")
    rx = (IX - cx) * DX
    ry = (IY - cy) * DY
    Q_2d = Q0 * np.exp(-(rx**2 + ry**2) / (2.0 * Q_SIGMA**2))  # (NX, NY)
    Q_3d = Q_2d[:, :, np.newaxis]  # (NX, NY, 1)

    sim = pkw.ThermalSimulation(
        nx=NX, ny=NY, nz=1,
        dx=DX, dy=DY, dz=DX,
        thermal_conductivity=K_TH,
        density=RHO,
        specific_heat=CP,
        enable_bioheat=True,
        perfusion_rate=WB_PER_S,
        blood_density=RHO_B,
        blood_specific_heat=CP_B,
        arterial_temperature=TA,
        initial_temperature=TA,
        track_thermal_dose=False,
        spatial_order=4,
    )

    print("  [pykwavers] Running simulation...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT, heat_source=Q_3d)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    # result.temperature is (NX, NY, 1) — squeeze to (NX, NY)
    T_py = np.asarray(result.temperature, dtype=np.float64)
    if T_py.ndim == 3:
        T_py = T_py[:, :, 0]
    return T_py


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(T_jl: np.ndarray, T_py: np.ndarray) -> None:
    """3-panel comparison: KWave.jl | pykwavers | difference."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    x_mm = np.arange(NX) * DX * 1e3 - NX * DX / 2 * 1e3
    y_mm = np.arange(NY) * DY * 1e3 - NY * DY / 2 * 1e3
    extent = [y_mm[0], y_mm[-1], x_mm[-1], x_mm[0]]

    # Remove body temperature offset so the colour map shows the temperature rise
    rise_jl = T_jl - TA
    rise_py = T_py - TA
    vmax = max(rise_jl.max(), rise_py.max())

    for ax, img, label in zip(
        axes[:2],
        [rise_jl, rise_py],
        ["KWave.jl", "pykwavers"],
    ):
        im = ax.imshow(
            img, extent=extent, aspect="equal", cmap="hot",
            vmin=0, vmax=vmax, origin="upper",
        )
        ax.set_title(f"{label}\n(T − {TA:.0f} °C)")
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("x [mm]")
        fig.colorbar(im, ax=ax, label="ΔT [°C]", fraction=0.046, pad=0.04)

    diff = rise_py - rise_jl
    dlim = max(np.abs(diff).max(), 1e-12)
    im = axes[2].imshow(
        diff, extent=extent, aspect="equal", cmap="RdBu_r",
        vmin=-dlim, vmax=dlim, origin="upper",
    )
    axes[2].set_title("pykwavers − KWave.jl")
    axes[2].set_xlabel("y [mm]")
    axes[2].set_ylabel("x [mm]")
    fig.colorbar(im, ax=axes[2], label="ΔΔT [°C]", fraction=0.046, pad=0.04)

    fig.suptitle(
        f"diff_homogeneous_medium_source_2d: KWave.jl vs pykwavers\n"
        f"Grid {NX}×{NY}  dx={DX*1e3:.2f} mm  σ={Q_SIGMA*1e3:.1f} mm  "
        f"Q0={Q0:.0e} W/m³  t={NT*DT:.0f} s",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D bioheat source: KWave.jl vs pykwavers parity check."
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Force KWave.jl re-run."
    )
    parser.add_argument(
        "--allow-failure", action="store_true",
        help="Exit 0 even if parity targets are not met.",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("diff_homogeneous_medium_source_2d: KWave.jl vs pykwavers")
    print(f"  Grid     : {NX}×{NY}   dx=dy={DX*1e3:.3f} mm  domain={NX*DX*1e3:.0f} mm")
    print(f"  Medium   : κ={K_TH}, ρ={RHO}, cp={CP}")
    print(f"  Perfusion: wb={WB_PER_S} 1/s  (jl: {WB_JL:.3f} kg/(m³·s))")
    print(f"  Source   : Q0={Q0:.0e} W/m³  σ={Q_SIGMA*1e3:.1f} mm")
    print(f"  Time     : dt={DT} s  Nt={NT}  t_end={NT*DT:.0f} s")
    print("=" * 62)

    print("\n[1/2] KWave.jl 2D diffusion...")
    T_jl = run_kwave_julia(force=args.no_cache)
    # Julia cx=nx÷2=64 (1-indexed) = 0-indexed cell 63 for NX=128
    jl_cx, jl_cy = NX // 2 - 1, NY // 2 - 1
    print(f"  KWave.jl: T_max={T_jl.max():.6f} °C  T_centre={T_jl[jl_cx, jl_cy]:.6f} °C")

    print("\n[2/2] pykwavers ThermalSimulation (nz=1 slab)...")
    T_py = run_pykwavers()
    print(f"  pykwavers: T_max={T_py.max():.6f} °C  T_centre={T_py[NX//2-1, NY//2-1]:.6f} °C")

    print("\n--- Parity evaluation ---")
    metrics = compute_image_metrics(T_jl, T_py)
    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"] >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
        "psnr_db":   metrics["psnr_db"]   >= thr["psnr_db"],
    }
    overall = "PASS" if all(checks.values()) else "FAIL"

    for key, ok in checks.items():
        val = metrics[key] if key != "rms_ratio" else metrics["rms_ratio"]
        print(f"  {key:12s} = {val:.6f}  {'[OK]' if ok else '[FAIL]'}")

    print(f"\n  Overall: {overall}")

    plot_comparison(T_jl, T_py)

    meta = {
        "engine_ref":  "KWave.jl/kwave_diffusion_2D",
        "engine_cand": "pykwavers.ThermalSimulation (nz=1)",
        "nx": NX, "ny": NY, "dx": DX, "dy": DY,
        "nt": NT, "dt": DT,
        "k_th": K_TH, "rho": RHO, "cp": CP,
        "wb_per_s": WB_PER_S, "wb_jl": WB_JL,
        "q0": Q0, "q_sigma_m": Q_SIGMA,
        "T_jl_max": float(T_jl.max()), "T_py_max": float(T_py.max()),
        "T_jl_centre": float(T_jl[NX//2-1, NY//2-1]),
        "T_py_centre": float(T_py[NX//2-1, NY//2-1]),
        **{k: float(v) for k, v in metrics.items()},
        "parity_status": overall,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    lines = [
        "diff_homogeneous_medium_source_2d_jl_compare",
        f"engine_ref     : KWave.jl/kwave_diffusion_2D",
        f"engine_cand    : pykwavers.ThermalSimulation (nz=1 slab)",
        f"nx, ny, dx, dt : {NX}, {NY}, {DX:.4e} m, {DT} s",
        f"k, rho, cp     : {K_TH}, {RHO}, {CP}",
        f"perfusion_jl   : {WB_JL:.4f} kg/(m³·s)  perfusion_pkw : {WB_PER_S:.6e} 1/s",
        f"Q_peak, sigma  : {Q0:.0e} W/m³, {Q_SIGMA*1e3:.1f} mm",
        f"T_jl(end)      : {T_jl.max():.6f} °C   T_py(end): {T_py.max():.6f} °C",
        f"T_jl_centre    : {T_jl[NX//2-1,NY//2-1]:.6f} °C",
        f"T_py_centre    : {T_py[NX//2-1,NY//2-1]:.6f} °C",
        f"pearson_r      : {metrics['pearson_r']:.6f}  (threshold >= {thr['pearson_r']})",
        f"rms_ratio      : {metrics['rms_ratio']:.6f}  (threshold {thr['rms_ratio_min']}-{thr['rms_ratio_max']})",
        f"psnr_db        : {metrics['psnr_db']:.3f}  (threshold >= {thr['psnr_db']} dB)",
        f"RESULT         : {overall}",
    ]
    save_text_report(METRICS_PATH, "", lines)
    print(f"  Saved: {METRICS_PATH}")

    if not all(checks.values()) and not args.allow_failure:
        raise SystemExit(
            "diff_homogeneous_medium_source_2d parity targets not met. "
            "Use --allow-failure to suppress exit code."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
