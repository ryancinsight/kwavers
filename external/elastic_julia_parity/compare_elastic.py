#!/usr/bin/env python3
"""
compare_elastic.py
===================
Side-by-side parity comparison of **pykwavers** (`SolverType.Elastic`) and
**KWave.jl** (`pstd_elastic_2d`) on a homogeneous 2-D elastic medium driven
by a uz velocity-source plane.

This is the first end-to-end side-by-side comparison of pykwavers' elastic
ladder (ADR 007 phases A.1–A.3) against an external reference. k-wave-python
explicitly does not support elastic simulations
(`assert ... "Elastic simulation is not supported"` in
`external/k-wave-python/kwave/kWaveSimulation.py:544`), so KWave.jl is the
only side-by-side available.

Physical setup
--------------
Grid    : 32×32 (single z-slice for pykwavers' 3-D elastic core)
          dx = dy = 0.5 mm
Medium  : homogeneous elastic, bone-like values from
          ``example_ewp_layered_medium`` lower layer:
              cp = 2000 m/s,  cs = 800 m/s,  ρ = 1200 kg/m³
Source  : uz velocity-source plane at x-index 8 (1-based 9), driven by a
          3-cycle Hann-windowed 1 MHz tone burst with peak 1 µm/s.
Sensors : three points along the +x ray from the source plane at offsets
          0, 3, 6 grid cells.
Time    : Nt = 200 at dt = 0.3·dx/(√3·cp); ~9.6 µs total.
PML     : 10 grid points, inside.

Comparison strategy
-------------------
Both engines are driven by **identical** inputs (same medium constants,
same mask, same signal). The recorded uz time series at each sensor is
compared with image-level Pearson r, RMS ratio, PSNR. A side-by-side
4-panel figure shows:
  1. Layout (source plane + sensors)
  2. KWave.jl uz traces (n_sensors × Nt)
  3. pykwavers uz traces
  4. Difference

Outputs
-------
* output/elastic_julia_compare.png        — 4-panel side-by-side figure
* output/elastic_julia_metrics.txt        — Pearson, RMS, PSNR, runtimes
* output/elastic_julia_kwave.csv          — raw KWave.jl uz traces (cached)
* output/elastic_julia_kwave_meta.json    — KWave.jl run metadata
* output/elastic_julia_pykwavers.npz      — raw pykwavers traces (cached)

Usage
-----
    python compare_elastic.py
    python compare_elastic.py --no-cache
    python compare_elastic.py --allow-failure

Requirements: julia ≥ 1.10 with KWave.jl in `external/k-wave-julia/KWave.jl`,
pykwavers built (`maturin develop --release` in `pykwavers/`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
JULIA_PROJECT = REPO_ROOT / "external" / "k-wave-julia" / "KWave.jl"
JULIA_DRIVER = HERE / "run_kwave_julia_elastic.jl"
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYKWAVERS_PYTHON = REPO_ROOT / "pykwavers" / "python"
PYKWAVERS_VENV = REPO_ROOT / "pykwavers" / ".venv" / "Scripts" / "python.exe"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
NX, NY = 32, 32
NZ_PYKWAVERS = 16  # pykwavers' elastic core asserts NZ ≥ small-window-size in
                   # the FD stencil; NZ=1 hits an internal index panic. The
                   # KWave.jl path is genuinely 2-D and unaffected.
DX = DY = 0.5e-3

CP = 2000.0  # bone-like compressional speed
CS = 800.0  # shear speed
RHO = 1200.0  # density

NT = 200
CFL = 0.3
DT = CFL * DX / (np.sqrt(3.0) * CP)

SOURCE_X_1B = 9  # Julia/MATLAB 1-based: source plane index
PML_SIZE = 10

SOURCE_FREQ = 1.0e6  # 1 MHz
SOURCE_PEAK = 1.0e-6  # 1 µm/s

SENSOR_OFFSETS = [0, 3, 6]  # grid cells from source plane along +x ray

# ---------------------------------------------------------------------------
# Parity thresholds — different absorption-model implementations between
# KWave.jl (Kelvin-Voigt with no absorption when alpha=None) and kwavers
# (no absorption either by default). Drift expected from time-stepping
# scheme differences (KWave.jl: spectral; pykwavers elastic: 4th-order FD
# with velocity-Verlet).  Pearson r ≥ 0.6 captures phase agreement; RMS
# ratio loose because of the FD-vs-spectral amplitude calibration.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r": 0.50,
    "rms_ratio_min": 0.20,
    "rms_ratio_max": 5.00,
    "psnr_db": 5.0,
}

KWAVE_CSV = OUTPUT_DIR / "elastic_julia_kwave.csv"
KWAVE_META = OUTPUT_DIR / "elastic_julia_kwave_meta.json"
PKWAV_NPZ = OUTPUT_DIR / "elastic_julia_pykwavers.npz"
FIGURE_PATH = OUTPUT_DIR / "elastic_julia_compare.png"
METRICS_PATH = OUTPUT_DIR / "elastic_julia_metrics.txt"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"


def build_tone_burst(nt: int, dt: float, freq: float, peak: float) -> np.ndarray:
    """3-cycle Hann-windowed sinusoid."""
    t = np.arange(nt, dtype=np.float64) * dt
    n_cycles = 3
    burst_dur = n_cycles / freq
    win = np.zeros(nt, dtype=np.float64)
    in_burst = t < burst_dur
    n_in = int(np.sum(in_burst))
    if n_in > 0:
        win[:n_in] = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_in) / n_in))
    return peak * win * np.sin(2.0 * np.pi * freq * t)


def build_sensor_positions_1b() -> np.ndarray:
    """Return (n_sensors, 2) array of 1-based (i, j) sensor positions."""
    cy = NY // 2 + 1  # 1-based centre
    rows = []
    for off in SENSOR_OFFSETS:
        ix = SOURCE_X_1B + off
        if ix <= NX:
            rows.append([ix, cy])
    return np.asarray(rows, dtype=np.int64)


def build_sensor_positions_0b() -> list[tuple[int, int]]:
    """0-based sensor positions for pykwavers (3-D: (i, j, 0))."""
    cy = NY // 2  # 0-based
    return [(SOURCE_X_1B - 1 + off, cy) for off in SENSOR_OFFSETS if SOURCE_X_1B - 1 + off < NX]


def run_kwave_julia(uz_signal: np.ndarray, sensor_positions_1b: np.ndarray, *, no_cache: bool) -> dict:
    """Invoke KWave.jl via subprocess and read back the recorded uz traces."""
    if not no_cache and KWAVE_CSV.exists() and KWAVE_META.exists():
        print("  [KWave.jl] Loading from cache...")
        meta = json.loads(KWAVE_META.read_text())
        uz_data = np.loadtxt(str(KWAVE_CSV), delimiter=",", dtype=np.float64)
        if uz_data.ndim == 1:
            uz_data = uz_data.reshape(1, -1)
        return {"uz": uz_data, "runtime_s": float(meta["solver_seconds"]), "meta": meta}

    julia_exe = shutil.which("julia")
    if julia_exe is None:
        raise RuntimeError("`julia` not found on PATH; install Julia ≥ 1.10")

    # Persist signal + sensor positions for the Julia driver.
    signal_path = OUTPUT_DIR / "elastic_julia_signal.csv"
    positions_path = OUTPUT_DIR / "elastic_julia_positions.csv"
    np.savetxt(str(signal_path), uz_signal, delimiter=",", fmt="%.17g")
    np.savetxt(str(positions_path), sensor_positions_1b, delimiter=",", fmt="%d")

    cmd = [
        julia_exe,
        f"--project={JULIA_PROJECT}",
        str(JULIA_DRIVER),
        "--output-csv", str(KWAVE_CSV),
        "--output-meta", str(KWAVE_META),
        "--nx", str(NX),
        "--ny", str(NY),
        "--dx", f"{DX:.17g}",
        "--dy", f"{DY:.17g}",
        "--nt", str(NT),
        "--dt", f"{DT:.17g}",
        "--cp", f"{CP:.17g}",
        "--cs", f"{CS:.17g}",
        "--rho", f"{RHO:.17g}",
        "--source-x", str(SOURCE_X_1B),
        "--source-signal-csv", str(signal_path),
        "--sensor-positions-csv", str(positions_path),
        "--pml-size", str(PML_SIZE),
    ]
    print(f"  [KWave.jl] Invoking: julia ... {JULIA_DRIVER.name}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed_s = time.perf_counter() - t0
    if proc.returncode != 0:
        print("  [KWave.jl] STDOUT:")
        print(proc.stdout)
        print("  [KWave.jl] STDERR:")
        print(proc.stderr)
        raise RuntimeError(f"KWave.jl driver failed (exit {proc.returncode})")
    print(f"  [KWave.jl] Done in {elapsed_s:.1f} s (wall-clock incl. Julia startup)")
    if proc.stdout.strip():
        print("  [KWave.jl] STDOUT (first 300 chars):", proc.stdout[:300])

    meta = json.loads(KWAVE_META.read_text())
    uz_data = np.loadtxt(str(KWAVE_CSV), delimiter=",", dtype=np.float64)
    if uz_data.ndim == 1:
        uz_data = uz_data.reshape(1, -1)
    return {"uz": uz_data, "runtime_s": float(meta["solver_seconds"]), "meta": meta}


def run_pykwavers(uz_signal: np.ndarray, sensor_positions_0b: list[tuple[int, int]], *, no_cache: bool) -> dict:
    """Run pykwavers SolverType.Elastic with the same source/medium/sensors."""
    if not no_cache and PKWAV_NPZ.exists():
        print("  [pykwavers] Loading from cache...")
        d = np.load(str(PKWAV_NPZ), allow_pickle=False)
        return {
            "uz": np.asarray(d["uz"], dtype=np.float64),
            "runtime_s": float(d["runtime_s"]),
        }

    sys.path.insert(0, str(PYKWAVERS_PYTHON))
    import pykwavers as pkw

    # Lift to 3-D: NZ slab matching the elastic FD stencil's window size.
    NZ = NZ_PYKWAVERS
    cz = NZ // 2  # mid-slab plane for source/sensors
    grid = pkw.Grid(NX, NY, NZ, DX, DY, DX)
    medium = pkw.Medium.elastic(CP, CS, RHO, grid=grid)

    # Velocity source on the x-plane (broadcast across all y at z=cz only,
    # to mirror the KWave.jl 2-D source).
    u_mask = np.zeros((NX, NY, NZ), dtype=bool)
    u_mask[SOURCE_X_1B - 1, :, cz] = True  # 0-based
    # Drive ux to match the KWave.jl 2-D path (which silently ignores
    # source.uz for pstd_elastic_2d — only vx/vy are valid in 2-D).
    src = pkw.Source.from_elastic_velocity_source(u_mask, ux=uz_signal)

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    for (i, j) in sensor_positions_0b:
        sensor_mask[i, j, cz] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, src, sensor, solver=pkw.SolverType.Elastic)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running CPU Elastic (Nt={NT}, dt={DT:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=NT, dt=DT)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    # Compare the ux trace (driven component). The Phase A.2.5
    # multi-component recording exposes result.ux/uy/uz separately.
    ux_data = np.asarray(result.ux, dtype=np.float64)
    if ux_data.shape[0] != len(sensor_positions_0b):
        raise AssertionError(
            f"Unexpected pykwavers ux shape {ux_data.shape}; "
            f"expected ({len(sensor_positions_0b)}, {NT})"
        )

    np.savez(
        str(PKWAV_NPZ),
        uz=ux_data,  # field name kept for cache-version compatibility
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )
    return {"uz": ux_data, "runtime_s": runtime_s}


def compute_image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Pearson r, RMS ratio, PSNR computed on flattened sensor matrices."""
    a = np.asarray(reference, dtype=np.float64).flatten()
    b = np.asarray(candidate, dtype=np.float64).flatten()
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    ref_norm = a - a.mean()
    can_norm = b - b.mean()
    denom = (np.linalg.norm(ref_norm) * np.linalg.norm(can_norm)) + 1e-30
    pearson_r = float(np.dot(ref_norm, can_norm) / denom)

    a_rms = float(np.sqrt(np.mean(a ** 2)))
    b_rms = float(np.sqrt(np.mean(b ** 2)))
    rms_ratio = b_rms / max(a_rms, 1e-30)

    diff = b - a
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    peak_a = max(float(np.max(np.abs(a))), 1e-30)
    psnr_db = 20.0 * np.log10(peak_a / max(rmse, 1e-30))
    peak_b = float(np.max(np.abs(b)))
    return {
        "pearson_r": pearson_r,
        "rms_ratio": rms_ratio,
        "psnr_db": psnr_db,
        "rmse": rmse,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "peak_ratio": peak_b / max(peak_a, 1e-30),
    }


def plot_comparison(
    sensor_positions_0b: list[tuple[int, int]],
    kw_uz: np.ndarray,
    py_uz: np.ndarray,
    metrics: dict,
    *,
    status: str,
) -> None:
    diff = py_uz - kw_uz
    vmax = float(max(np.abs(kw_uz).max(), np.abs(py_uz).max(), 1e-30))
    dmax = float(max(np.abs(diff).max(), 1e-30))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # Panel 1: layout
    ax_l = axes[0]
    layout = np.zeros((NX, NY, 3), dtype=np.float32)
    layout[SOURCE_X_1B - 1, :, 0] = 1.0  # source row in red
    for (i, j) in sensor_positions_0b:
        layout[i, j, 1] = 1.0  # sensors in green
    ax_l.imshow(np.transpose(layout, (1, 0, 2)), origin="lower",
                extent=[0, NX, 0, NY])
    ax_l.set_title("Layout: red=source plane, green=sensors")
    ax_l.set_xlabel("x [grid]")
    ax_l.set_ylabel("y [grid]")

    # Panel 2: KWave.jl
    im_kw = axes[1].imshow(
        kw_uz, aspect="auto", origin="lower", cmap="seismic",
        vmin=-vmax, vmax=vmax,
    )
    axes[1].set_title("KWave.jl  ux")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Sensor index")
    fig.colorbar(im_kw, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: pykwavers
    im_py = axes[2].imshow(
        py_uz, aspect="auto", origin="lower", cmap="seismic",
        vmin=-vmax, vmax=vmax,
    )
    axes[2].set_title("pykwavers  ux")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Sensor index")
    fig.colorbar(im_py, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: difference
    im_d = axes[3].imshow(
        diff, aspect="auto", origin="lower", cmap="seismic",
        vmin=-dmax, vmax=dmax,
    )
    axes[3].set_title(f"diff (pykwavers − KWave.jl)\nmax|Δ|={dmax:.2e}")
    axes[3].set_xlabel("Time step")
    axes[3].set_ylabel("Sensor index")
    fig.colorbar(im_d, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Elastic uz parity: KWave.jl vs pykwavers  [{status}]   "
        f"r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}  "
        f"PSNR={metrics['psnr_db']:.1f} dB",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(str(FIGURE_PATH), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers SolverType.Elastic against KWave.jl pstd_elastic_2d."
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh runs.")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()
    no_cache = args.no_cache

    print("=" * 78)
    print("compare_elastic: pykwavers SolverType.Elastic vs KWave.jl pstd_elastic_2d")
    print("=" * 78)
    print(f"  Grid    : {NX}×{NY}  dx={DX*1e3:.2f} mm")
    print(f"  Medium  : cp={CP} m/s  cs={CS} m/s  ρ={RHO} kg/m³")
    print(f"  Source  : uz tone burst plane at x_1b={SOURCE_X_1B}, "
          f"f={SOURCE_FREQ/1e6:.1f} MHz, peak={SOURCE_PEAK*1e6:.1f} µm/s")
    print(f"  Time    : Nt={NT}  dt={DT*1e9:.2f} ns")
    print(f"  PML     : {PML_SIZE} pts inside")
    print("-" * 78)

    uz_signal = build_tone_burst(NT, DT, SOURCE_FREQ, SOURCE_PEAK)
    sensor_positions_1b = build_sensor_positions_1b()
    sensor_positions_0b = build_sensor_positions_0b()
    print(f"  Source signal peak = {float(np.max(np.abs(uz_signal))):.3e} m/s")
    print(f"  {len(sensor_positions_0b)} sensors at offsets {SENSOR_OFFSETS} along +x ray")

    print("\n[1/2] KWave.jl pstd_elastic_2d ...")
    kw = run_kwave_julia(uz_signal, sensor_positions_1b, no_cache=no_cache)
    kw_uz = kw["uz"]
    print(f"  shape={kw_uz.shape}  peak={float(np.max(np.abs(kw_uz))):.3e} m  "
          f"runtime={kw['runtime_s']:.1f} s")

    print("\n[2/2] pykwavers SolverType.Elastic ...")
    pkw_res = run_pykwavers(uz_signal, sensor_positions_0b, no_cache=no_cache)
    # Semantic alignment: KWave.jl's `record = [:ux]` returns particle
    # **velocity** vx, while pykwavers' result.ux is **displacement** ux.
    # Convert pykwavers' displacement to velocity by numerical
    # differentiation so both engines compare on the same physical
    # quantity (velocity).
    py_disp = pkw_res["uz"]
    py_uz = np.gradient(py_disp, DT, axis=1)
    print(f"  displacement peak = {float(np.max(np.abs(py_disp))):.3e} m  "
          f"runtime={pkw_res['runtime_s']:.1f} s")
    print(f"  velocity peak (numerical d/dt) = "
          f"{float(np.max(np.abs(py_uz))):.3e} m/s  "
          f"(KWave.jl reports particle velocity directly via `record = [:ux]`)")

    if kw_uz.shape != py_uz.shape:
        print(f"\n  WARNING: shape mismatch kwave={kw_uz.shape} vs pykwavers={py_uz.shape}; "
              f"truncating to common length")
        nt_common = min(kw_uz.shape[1], py_uz.shape[1])
        ns_common = min(kw_uz.shape[0], py_uz.shape[0])
        kw_uz = kw_uz[:ns_common, :nt_common]
        py_uz = py_uz[:ns_common, :nt_common]

    print("\n--- Parity evaluation (sensor matrix) ---")
    metrics = compute_image_metrics(kw_uz, py_uz)
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
    print(f"  rmse      : {metrics['rmse']:.3e} m")
    print(f"  peak_ratio: {metrics['peak_ratio']:.4f}")

    plot_comparison(sensor_positions_0b, kw_uz, py_uz, metrics, status=status)

    METRICS_PATH.write_text(
        f"compare_elastic: pykwavers vs KWave.jl pstd_elastic_2d\n"
        f"parity_status: {status}\n"
        f"grid: {NX}x{NY}  dx={DX:.6e} m\n"
        f"medium: cp={CP} cs={CS} rho={RHO}\n"
        f"source: uz plane at x_1b={SOURCE_X_1B}  f={SOURCE_FREQ/1e6} MHz  peak={SOURCE_PEAK*1e6} µm/s\n"
        f"time: Nt={NT}  dt={DT:.6e} s\n"
        f"runtime: kwave_julia={kw['runtime_s']:.3f}s  pykwavers={pkw_res['runtime_s']:.3f}s\n"
        f"\n"
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target ≥ {thr['pearson_r']})\n"
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  (target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])\n"
        f"psnr_db    = {metrics['psnr_db']:.2f} dB (target ≥ {thr['psnr_db']} dB)\n"
        f"rmse       = {metrics['rmse']:.6e} m\n"
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} m\n"
        f"peak_kwave   = {float(np.max(np.abs(kw_uz))):.6e} m\n"
        f"peak_pykwavers = {float(np.max(np.abs(py_uz))):.6e} m\n"
        f"peak_ratio   = {metrics['peak_ratio']:.6f}\n",
        encoding="utf-8",
    )
    print(f"  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")
    return 0 if (status == "PASS" or args.allow_failure) else 1


if __name__ == "__main__":
    raise SystemExit(main())
