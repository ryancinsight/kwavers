#!/usr/bin/env python3
"""
checkpointing_compare.py
========================
Validates pykwavers PSTD checkpoint save/resume against an uninterrupted
reference run.

Theorem: Bit-exact continuation
    For a PSTD simulation of Nt steps split at step k:

        run_to_checkpoint(steps=k) -> save(checkpoint_path)
        run_from_checkpoint(checkpoint_path) -> resume for Nt-k steps

    The sensor time series is bit-exact (float64 equality) to running Nt steps
    in a single call because:
      - the checkpoint stores the seven primary field arrays as raw IEEE-754
        double-precision values;
      - all spectral operators are deterministically recomputed from the fixed
        grid and PSTDConfig at solver construction time;
      - the sensor recorder state before the checkpoint is restored verbatim;
      - the restored solver continues with the same binary and input data.

Grid:   32×32×32, dx=1 mm
Medium: water (c=1500 m/s, rho=1000 kg/m³, lossless)
Source: initial pressure Gaussian pulse at grid centre
Sensor: full grid (all points)
Steps:  Nt=40 (split at k=20)

Outputs
-------
pykwavers/examples/output/checkpointing_compare.png   — sensor-point overlay
pykwavers/examples/output/checkpointing_metrics.txt    — exact parity report
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import DEFAULT_OUTPUT_DIR, bootstrap_example_paths, save_text_report


_ROOT = bootstrap_example_paths()

try:
    import pykwavers as pkw
except ImportError as exc:
    sys.exit(f"pykwavers not installed — run: cd pykwavers && maturin develop --release ({exc})")


NX, NY, NZ = 32, 32, 32
DX = 1e-3
C0 = 1500.0
RHO0 = 1000.0
NT = 40
SPLIT = NT // 2
DT = 0.3 * DX / (C0 * 3.0 ** 0.5)

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
FIGURE_PATH = OUTPUT_DIR / "checkpointing_compare.png"
METRICS_PATH = OUTPUT_DIR / "checkpointing_metrics.txt"


def _build_problem() -> tuple[object, object, object, object]:
    grid = pkw.Grid(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DX, dz=DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    cx, cy, cz = NX / 2, NY / 2, NZ / 2
    sigma = 3.0 * DX
    xx, yy, zz = np.meshgrid(
        np.arange(NX) * DX,
        np.arange(NY) * DX,
        np.arange(NZ) * DX,
        indexing="ij",
    )
    p0 = np.exp(
        -((xx - cx * DX) ** 2 + (yy - cy * DX) ** 2 + (zz - cz * DX) ** 2)
        / (2.0 * sigma**2)
    ).astype(np.float64)

    source = pkw.Source.from_initial_pressure(p0)
    sensor = pkw.Sensor.grid()
    return grid, medium, source, sensor


def _run_reference(grid, medium, source, sensor) -> dict[str, object]:
    t0 = time.perf_counter()
    simulation = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    result = simulation.run(time_steps=NT, dt=DT)
    runtime_s = time.perf_counter() - t0
    sensor_data = np.asarray(result.sensor_data)
    return {
        "sensor_data": sensor_data,
        "runtime_s": runtime_s,
        "shape": tuple(sensor_data.shape),
    }


def _run_checkpointed(grid, medium, source, sensor) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="kwavers_checkpointing_") as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "checkpoint.bin"

        t0 = time.perf_counter()
        checkpoint_sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
        checkpoint_sim.run_to_checkpoint(
            time_steps=NT,
            checkpoint_steps=SPLIT,
            checkpoint_path=str(checkpoint_path),
            dt=DT,
        )
        checkpoint_runtime_s = time.perf_counter() - t0
        checkpoint_size_bytes = checkpoint_path.stat().st_size

        t0 = time.perf_counter()
        resume_sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
        resumed = resume_sim.run_from_checkpoint(
            time_steps=NT,
            checkpoint_path=str(checkpoint_path),
            dt=DT,
        )
        resume_runtime_s = time.perf_counter() - t0

        resumed_data = np.asarray(resumed.sensor_data)
        checkpoint_deleted = not checkpoint_path.exists()

        return {
            "sensor_data": resumed_data,
            "shape": tuple(resumed_data.shape),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "checkpoint_runtime_s": checkpoint_runtime_s,
            "resume_runtime_s": resume_runtime_s,
            "checkpoint_deleted": checkpoint_deleted,
        }


def _save_plot(reference_data: np.ndarray, resumed_data: np.ndarray, max_abs_error: float, bit_exact: bool) -> None:
    if reference_data.ndim == 1:
        ref_trace = reference_data
        res_trace = resumed_data
    else:
        centre_flat = (NX // 2) * NY * NZ + (NY // 2) * NZ + NZ // 2
        if centre_flat < reference_data.shape[0]:
            ref_trace = reference_data[centre_flat, :]
            res_trace = resumed_data[centre_flat, :]
        else:
            ref_trace = reference_data[0, :]
            res_trace = resumed_data[0, :]

    t_arr = np.arange(NT) * DT * 1e9

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    ax0 = axes[0]
    ax0.plot(t_arr, ref_trace, "b-", linewidth=1.5, label="Reference (no checkpoint)")
    ax0.plot(t_arr, res_trace, "r--", linewidth=1.0, label="Resumed (from checkpoint)")
    ax0.set_xlabel("Time (ns)")
    ax0.set_ylabel("Pressure (Pa)")
    ax0.set_title(f"Sensor trace — centre voxel  [Nt={NT}, split at k={SPLIT}]")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(t_arr, ref_trace - res_trace, "k-", linewidth=1.0)
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Error (Pa)")
    ax1.set_title(f"Reference − Resumed  [max={max_abs_error:.2e}  bit-exact={bit_exact}]")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def build_report_lines(result: dict[str, object]) -> list[str]:
    reference = result["reference"]
    checkpoint = result["checkpoint"]
    metrics = result["metrics"]
    return [
        f"checkpoint_status: {result['status']}",
        f"reference_runtime_s: {reference['runtime_s']:.6f}",
        f"checkpoint_runtime_s: {checkpoint['checkpoint_runtime_s']:.6f}",
        f"resume_runtime_s: {checkpoint['resume_runtime_s']:.6f}",
        f"checkpoint_size_bytes: {checkpoint['checkpoint_size_bytes']}",
        f"checkpoint_deleted: {checkpoint['checkpoint_deleted']}",
        f"max_absolute_error: {metrics['max_absolute_error']:.6e}",
        f"bit_exact: {metrics['bit_exact']}",
        f"reference_shape: {metrics['reference_shape']}",
        f"resumed_shape: {metrics['resumed_shape']}",
        f"nt: {NT}",
        f"split_step: {SPLIT}",
    ]


def run_comparison() -> dict[str, object]:
    grid, medium, source, sensor = _build_problem()
    reference = _run_reference(grid, medium, source, sensor)
    checkpoint = _run_checkpointed(grid, medium, source, sensor)

    reference_data = np.asarray(reference["sensor_data"], dtype=np.float64)
    resumed_data = np.asarray(checkpoint["sensor_data"], dtype=np.float64)

    if reference_data.shape != resumed_data.shape:
        max_abs_error = float("inf")
        bit_exact = False
    else:
        max_abs_error = float(np.max(np.abs(reference_data - resumed_data)))
        bit_exact = bool(
            np.array_equal(reference_data.view(np.uint64), resumed_data.view(np.uint64))
        )

    status = "PASS" if bit_exact and checkpoint["checkpoint_deleted"] else "FAIL"
    result: dict[str, object] = {
        "status": status,
        "reference": {
            "runtime_s": reference["runtime_s"],
            "shape": reference["shape"],
        },
        "checkpoint": {
            "checkpoint_runtime_s": checkpoint["checkpoint_runtime_s"],
            "resume_runtime_s": checkpoint["resume_runtime_s"],
            "checkpoint_path": checkpoint["checkpoint_path"],
            "checkpoint_size_bytes": checkpoint["checkpoint_size_bytes"],
            "checkpoint_deleted": checkpoint["checkpoint_deleted"],
        },
        "resumed": {
            "shape": checkpoint["shape"],
        },
        "metrics": {
            "max_absolute_error": max_abs_error,
            "bit_exact": bit_exact,
            "reference_shape": reference["shape"],
            "resumed_shape": checkpoint["shape"],
            "nt": NT,
            "split_step": SPLIT,
        },
        "plot_path": str(FIGURE_PATH),
        "metrics_path": str(METRICS_PATH),
    }

    save_text_report(METRICS_PATH, "checkpointing parity report", build_report_lines(result))
    _save_plot(reference_data, resumed_data, max_abs_error, bit_exact)
    return result


def main() -> int:
    result = run_comparison()
    metrics = result["metrics"]
    print(f"checkpoint_status: {result['status']}")
    print(f"parity_status: {result['status']}")
    print(f"reference_shape: {metrics['reference_shape']}")
    print(f"resumed_shape:   {metrics['resumed_shape']}")
    print(f"max_absolute_error: {metrics['max_absolute_error']:.6e}")
    print(f"bit_exact: {metrics['bit_exact']}")
    print(f"checkpoint_deleted: {result['checkpoint']['checkpoint_deleted']}")
    print(f"Saved {METRICS_PATH}")
    print(f"Saved {FIGURE_PATH}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
