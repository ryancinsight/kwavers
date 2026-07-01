#!/usr/bin/env python3
"""
pr_time_reversal_2d_jl_compare.py
=================================
KWave.jl photoacoustic time-reversal vs pykwavers
``time_reversal_reconstruction`` for the same 2-D forward sensor data.

Why this script exists
----------------------
``external/k-wave-python/examples/pr_2D_FFT_line_sensor.py`` covers
*FFT-based* line-sensor reconstruction; KWave.jl publishes a
*time-reversal*-based variant in ``examples/pr_time_reversal_2d.jl``
which has no k-wave-python equivalent. This script wires the two engines
together to validate the recon-pipeline parity.

Test setup
----------
Forward propagation is run on the KWave.jl side (deterministic spectral
solver). The recorded line-sensor pressure matrix is then reconstructed
by BOTH engines:

    * KWave.jl: native ``kspace_first_order`` with
      ``time_reversal_boundary_data``.
    * pykwavers: ``time_reversal_reconstruction(pressure, positions, ...)``.

Comparing the two reconstructions isolates the recon path from any
forward-engine differences. Both should converge to the same image of the
two p0 disc sources up to PML boundary effects.

Parity criteria:
    Pearson r (recon vs recon)         >= 0.60
    peak_ratio (recon vs recon)        in [0.7, 1.4]
    Pearson r (each recon vs p0 truth) >= 0.35  (TR is intrinsically lossy
                                                 with a single line sensor)

Outputs:
    output/pr_time_reversal_2d_jl_compare.png
    output/pr_time_reversal_2d_jl_metrics.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
JULIA_DRIVER = HERE / "run_kwave_julia_pr_time_reversal_2d.jl"

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH = OUTPUT_DIR / "pr_time_reversal_2d_jl_compare.png"
METRICS_PATH = OUTPUT_DIR / "pr_time_reversal_2d_jl_metrics.txt"
JL_SENSOR_CSV = OUTPUT_DIR / "pr_time_reversal_2d_jl_sensor.csv"
JL_RECON_CSV = OUTPUT_DIR / "pr_time_reversal_2d_jl_recon.csv"
JL_P0_CSV = OUTPUT_DIR / "pr_time_reversal_2d_jl_p0.csv"
JL_META = OUTPUT_DIR / "pr_time_reversal_2d_jl_meta.json"

# ---------------------------------------------------------------------------
# Parameters (kept small so the run completes in <60 s on either engine)
# ---------------------------------------------------------------------------
NX, NY = 64, 64
DX = DY = 0.1e-3                  # 0.1 mm
C0 = 1500.0
RHO0 = 1000.0
SENSOR_X_1BASED = 1               # left edge sensor line
PML_SIZE = 20

PARITY_THRESHOLDS = {
    # Recon-vs-recon Pearson is intrinsically degraded by the differing
    # internal TR formulations (kspace_first_order TR re-injection vs
    # pykwavers' forward-then-time-reverse implementation), as well as
    # FD-vs-spectral truncation differences observed in the bioheat parity.
    # We require strong but not bit-exact agreement.
    "r_recon_vs_recon":    0.60,
    "peak_recon_vs_recon": (0.70, 1.40),
    # Single-line-sensor TR cannot fully reconstruct a 2-D source; both
    # engines are independently expected to land in r ≈ 0.4–0.6 vs truth.
    "r_recon_vs_p0":       0.35,
}


def run_julia() -> dict:
    julia = os.environ.get("JULIA_BIN", "julia")
    cmd = [
        julia, f"--project={JULIA_PROJECT}", str(JULIA_DRIVER),
        "--nx", str(NX), "--ny", str(NY),
        "--dx", str(DX), "--dy", str(DY),
        "--c0", str(C0), "--rho0", str(RHO0),
        "--sensor-x-1based", str(SENSOR_X_1BASED),
        "--pml-size", str(PML_SIZE),
        "--out-sensor-csv", str(JL_SENSOR_CSV),
        "--out-recon-csv",  str(JL_RECON_CSV),
        "--out-p0-csv",     str(JL_P0_CSV),
        "--out-meta",       str(JL_META),
    ]
    print("[julia] launching:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Julia driver failed (exit {proc.returncode})")

    sensor = np.loadtxt(JL_SENSOR_CSV, delimiter=",")     # (n_sens, nt)
    recon = np.loadtxt(JL_RECON_CSV, delimiter=",").T     # (nx, ny)
    p0 = np.loadtxt(JL_P0_CSV, delimiter=",").T           # (nx, ny)
    meta = json.loads(JL_META.read_text())
    return {"sensor": sensor, "recon": recon, "p0": p0, "meta": meta}


def run_pykwavers_recon(sensor_pressure: np.ndarray, dt: float) -> np.ndarray:
    """Reconstruct via pykwavers ``time_reversal_reconstruction``."""
    n_sens = sensor_pressure.shape[0]
    # Sensor positions in metres — line at i=SENSOR_X_1BASED-1 (0-based) for
    # all j. Match the column-major order KWave.jl uses for its sensor mask.
    sensor_positions = np.zeros((n_sens, 3), dtype=np.float64)
    sx = (SENSOR_X_1BASED - 1) * DX
    for k in range(n_sens):
        sensor_positions[k, 0] = sx
        sensor_positions[k, 1] = k * DY
        sensor_positions[k, 2] = 0.0

    grid = pkw.Grid(
        nx=NX, ny=NY, nz=1,
        dx=DX, dy=DY, dz=DX,
    )
    recon = pkw.time_reversal_reconstruction(
        np.asarray(sensor_pressure, dtype=np.float64),
        sensor_positions,
        grid,
        float(C0),
        float(1.0 / dt),
        pml_size=PML_SIZE,
    )
    return np.asarray(recon, dtype=np.float64).squeeze()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    jl = run_julia()
    dt = float(jl["meta"]["dt"])
    print(f"[meta] nt={jl['meta']['nt']}, dt={dt:.3e} s")

    py_recon = run_pykwavers_recon(jl["sensor"], dt)

    # Both reconstructions should be (NX, NY). pykwavers may return (NX, NY, 1)
    # or (NX, NY); shape-normalise.
    py_recon = py_recon.reshape(NX, NY) if py_recon.size == NX * NY else py_recon
    jl_recon = jl["recon"]
    p0 = jl["p0"]

    # Crop the PML border before metric computation — recon energy near the
    # boundary is meaningless on either engine.
    pad = PML_SIZE + 2
    s = (slice(pad, NX - pad), slice(pad, NY - pad))
    jl_inner = jl_recon[s]
    py_inner = py_recon[s]
    p0_inner = p0[s]

    m_recon = compute_image_metrics(jl_inner, py_inner)
    m_jl_p0 = compute_image_metrics(p0_inner, jl_inner)
    m_py_p0 = compute_image_metrics(p0_inner, py_inner)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    vmax = float(max(np.max(np.abs(jl_recon)), np.max(np.abs(py_recon)),
                     np.max(np.abs(p0))))

    axes[0, 0].imshow(p0.T, origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
    axes[0, 0].set_title("Source p0 (truth)")
    axes[0, 1].imshow(jl["sensor"], aspect="auto", cmap="seismic",
                       vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title(f"Line sensor data ({jl['sensor'].shape[0]} sensors)")
    axes[0, 1].set_xlabel("time step"); axes[0, 1].set_ylabel("sensor")

    axes[1, 0].imshow(jl_recon.T, origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
    axes[1, 0].set_title(
        f"KWave.jl TR recon  (r vs p0={m_jl_p0['pearson_r']:.3f})"
    )
    axes[1, 1].imshow(py_recon.T, origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
    axes[1, 1].set_title(
        f"pykwavers TR recon (r vs p0={m_py_p0['pearson_r']:.3f}, "
        f"r vs jl={m_recon['pearson_r']:.3f})"
    )
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=140); plt.close(fig)

    pmin, pmax = PARITY_THRESHOLDS["peak_recon_vs_recon"]
    pass_fail = (
        m_recon["pearson_r"] >= PARITY_THRESHOLDS["r_recon_vs_recon"]
        and pmin <= m_recon["peak_ratio"] <= pmax
        and m_jl_p0["pearson_r"] >= PARITY_THRESHOLDS["r_recon_vs_p0"]
        and m_py_p0["pearson_r"] >= PARITY_THRESHOLDS["r_recon_vs_p0"]
    )

    lines = [
        f"engine_ref (recon)  : KWave.jl/kspace_first_order TR",
        f"engine_cand (recon) : pykwavers.time_reversal_reconstruction",
        f"sensor data source  : KWave.jl forward kspace_first_order",
        f"nx,ny,dx            : {NX},{NY},{DX}",
        f"c0,rho0,nt,dt       : {C0},{RHO0},{jl['meta']['nt']},{dt:.3e}",
        f"sensor_line         : i={SENSOR_X_1BASED} (1-based), all j",
        f"pml_size            : {PML_SIZE}",
        f"-- recon vs recon --",
        f"  pearson_r         : {m_recon['pearson_r']:.4f}  "
        f"(threshold >= {PARITY_THRESHOLDS['r_recon_vs_recon']})",
        f"  peak_ratio        : {m_recon['peak_ratio']:.4f}  "
        f"(band [{pmin}, {pmax}])",
        f"  rms_ratio         : {m_recon['rms_ratio']:.4f}",
        f"-- recon vs p0 truth --",
        f"  KWave.jl    : r={m_jl_p0['pearson_r']:.4f}  "
        f"peak_ratio={m_jl_p0['peak_ratio']:.4f}",
        f"  pykwavers   : r={m_py_p0['pearson_r']:.4f}  "
        f"peak_ratio={m_py_p0['peak_ratio']:.4f}",
        f"RESULT              : {'PASS' if pass_fail else 'FAIL'}",
    ]
    save_text_report(METRICS_PATH, "pr_time_reversal_2d_jl_compare", lines)
    print("\n".join(lines))
    print(f"\nFigure : {FIGURE_PATH}")
    print(f"Metrics: {METRICS_PATH}")

    if not pass_fail and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
