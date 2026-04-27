#!/usr/bin/env python3
"""
Parity comparison for the vendored k-wave-python ``at_linear_array_transducer`` example.

This example reuses the repaired rotated-rectangle geometry path in pykwavers and
drives the native solver with the distributed per-point source matrix that k-Wave
would generate from the same element delays. The comparison reports:

* binary source mask overlap,
* weighted source-mask mass agreement,
* on-sensor p_max field parity.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


_ROOT = bootstrap_example_paths()

import pykwavers as pkw
from linear_array_transducer_geometry import (
    DX,
    NX,
    NY,
    NZ,
    build_kwave_array,
    build_pykwavers_array,
    build_pykwavers_source_matrix_and_masks,
    build_sensor_mask,
    build_source_signal_matrix,
    C0,
    SOURCE_AMP,
    SOURCE_CYCLES,
    SOURCE_F0,
)
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


GRID_POINTS = (NX, NY, NZ)
GRID_SPACING = (DX, DX, DX)
PML_SIZE = 10
T_END = 35e-6
CFL = 0.5
SOURCE_MODE = "additive"
COMPATIBILITY_MODE = "reference"

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "at_linear_array_transducer_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "at_linear_array_transducer_metrics.txt"
KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "at_linear_array_transducer_kwave_cache.npz"
PYKWAVERS_CACHE = DEFAULT_OUTPUT_DIR / "at_linear_array_transducer_pykwavers_cache.npz"
CACHE_VERSION = 5
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"


def _load_cached_result(path: Path) -> dict[str, np.ndarray | float] | None:
    if REFRESH_CACHE or not path.exists():
        return None
    cached = np.load(path, allow_pickle=False)
    version = int(np.asarray(cached["cache_version"]).reshape(())) if "cache_version" in cached.files else 0
    if version != CACHE_VERSION:
        return None
    return {
        "p_max": np.asarray(cached["p_max"], dtype=np.float64),
        "source_binary_mask": np.asarray(cached["source_binary_mask"], dtype=bool),
        "source_weighted_mask": np.asarray(cached["source_weighted_mask"], dtype=np.float64),
        "runtime_s": float(cached["runtime_s"]),
        "dt": float(cached["dt"]),
        "time_steps": int(np.asarray(cached["time_steps"]).reshape(())),
    }


def _build_kwave_configuration() -> tuple[kWaveGrid, kWaveMedium, kSource, kSensor, np.ndarray, np.ndarray]:
    kgrid = kWaveGrid(GRID_POINTS, GRID_SPACING)
    kgrid.makeTime(C0, CFL, T_END)
    medium = kWaveMedium(sound_speed=C0, density=1000.0)

    karray = build_kwave_array(kgrid)
    source_signal = build_source_signal_matrix(kgrid)
    source_binary_mask = np.asarray(karray.get_array_binary_mask(kgrid), dtype=bool)
    source_weighted_mask = np.asarray(karray.get_array_grid_weights(kgrid), dtype=np.float64)
    distributed_signal = np.asarray(karray.get_distributed_source_signal(kgrid, source_signal), dtype=np.float64)

    source = kSource()
    source.p_mask = source_binary_mask
    source.p = distributed_signal

    sensor = kSensor()
    sensor.mask = build_sensor_mask(NX, NY, NZ)
    sensor.record = ["p_max"]
    return kgrid, medium, source, sensor, source_binary_mask, source_weighted_mask


def _build_pykwavers_configuration() -> tuple[pkw.Grid, pkw.Medium, pkw.Source, pkw.Sensor, np.ndarray, np.ndarray]:
    grid = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=1000.0)

    kgrid = kWaveGrid(GRID_POINTS, GRID_SPACING)
    kgrid.makeTime(C0, CFL, T_END)
    source_signal = build_source_signal_matrix(kgrid)

    source_binary_mask, source_weighted_mask, distributed_signal, _, _ = build_pykwavers_source_matrix_and_masks(
        grid, source_signal
    )
    source = pkw.Source.from_mask(
        source_binary_mask.astype(np.float64),
        distributed_signal,
        SOURCE_F0,
        mode=SOURCE_MODE,
    )

    sensor = pkw.Sensor.from_mask(build_sensor_mask(NX, NY, NZ))
    sensor.set_record(["p_max"])
    return grid, medium, source, sensor, source_binary_mask, source_weighted_mask


def run_kwave_reference() -> dict[str, np.ndarray | float]:
    cached = _load_cached_result(KWAVE_CACHE)
    if cached is not None:
        return cached

    kgrid, medium, source, sensor, source_binary_mask, source_weighted_mask = _build_kwave_configuration()
    sim_opts = SimulationOptions(
        pml_inside=False,
        pml_size=PML_SIZE,
        data_cast="single",
        save_to_disk=True,
    )
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=True)

    start = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    runtime_s = time.perf_counter() - start

    p_max = np.asarray(sensor_data["p_max"], dtype=np.float64)
    if p_max.shape != (NX, NZ) and p_max.T.shape == (NX, NZ):
        p_max = p_max.T
    if p_max.shape != (NX, NZ):
        p_max = np.reshape(p_max, (NX, NZ), order="F")

    output = {
        "p_max": p_max,
        "source_binary_mask": source_binary_mask,
        "source_weighted_mask": source_weighted_mask,
        "runtime_s": runtime_s,
        "dt": float(kgrid.dt),
        "time_steps": int(kgrid.Nt),
    }
    np.savez(KWAVE_CACHE, cache_version=CACHE_VERSION, **output)
    return output


def run_pykwavers_reference() -> dict[str, np.ndarray | float]:
    cached = _load_cached_result(PYKWAVERS_CACHE)
    if cached is not None:
        return cached

    grid, medium, source, sensor, source_binary_mask, source_weighted_mask = _build_pykwavers_configuration()
    kgrid = kWaveGrid(GRID_POINTS, GRID_SPACING)
    kgrid.makeTime(C0, CFL, T_END)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(False)
    sim.set_compatibility_mode(COMPATIBILITY_MODE)

    start = time.perf_counter()
    result = sim.run(time_steps=int(kgrid.Nt), dt=float(kgrid.dt))
    runtime_s = time.perf_counter() - start

    p_max = np.asarray(result.p_max, dtype=np.float64)
    if p_max.shape != (NX, NZ) and p_max.T.shape == (NX, NZ):
        p_max = p_max.T
    if p_max.shape != (NX, NZ):
        p_max = np.reshape(p_max, (NX, NZ), order="F")

    output = {
        "p_max": p_max,
        "source_binary_mask": source_binary_mask,
        "source_weighted_mask": source_weighted_mask,
        "runtime_s": runtime_s,
        "dt": float(kgrid.dt),
        "time_steps": int(kgrid.Nt),
    }
    np.savez(PYKWAVERS_CACHE, cache_version=CACHE_VERSION, **output)
    return output


def _mask_overlap_metrics(kw_mask: np.ndarray, py_mask: np.ndarray) -> dict[str, float]:
    kw_mask = np.asarray(kw_mask, dtype=bool)
    py_mask = np.asarray(py_mask, dtype=bool)
    intersection = int(np.logical_and(kw_mask, py_mask).sum())
    union = int(np.logical_or(kw_mask, py_mask).sum())
    kw_count = int(kw_mask.sum())
    py_count = int(py_mask.sum())
    return {
        "kwave_active_cells": float(kw_count),
        "pykwavers_active_cells": float(py_count),
        "active_cell_ratio": float(py_count / kw_count) if kw_count else float("inf"),
        "intersection": float(intersection),
        "iou": float(intersection / union) if union else 0.0,
        "dice": float(2 * intersection / (kw_count + py_count)) if (kw_count + py_count) else 0.0,
    }


def run_comparison() -> dict[str, object]:
    kw = run_kwave_reference()
    py = run_pykwavers_reference()

    source_mask_metrics = _mask_overlap_metrics(kw["source_binary_mask"], py["source_binary_mask"])
    source_weighted_metrics = compute_image_metrics(kw["source_weighted_mask"], py["source_weighted_mask"])
    p_max_metrics = compute_image_metrics(kw["p_max"], py["p_max"])

    return {
        "kwave": kw,
        "pykwavers": py,
        "metrics": {
            "source_mask": source_mask_metrics,
            "source_weighted_mask": source_weighted_metrics,
            "p_max": p_max_metrics,
        },
    }


def save_comparison_figure(result: dict[str, object], figure_path: Path) -> None:
    kw_p_max = np.asarray(result["kwave"]["p_max"], dtype=np.float64)
    py_p_max = np.asarray(result["pykwavers"]["p_max"], dtype=np.float64)
    diff = np.abs(kw_p_max - py_p_max)

    vmin = min(float(kw_p_max.min()), float(py_p_max.min()))
    vmax = max(float(kw_p_max.max()), float(py_p_max.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    im0 = axes[0].imshow(kw_p_max.T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("k-wave-python p_max")
    axes[1].imshow(py_p_max.T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("pykwavers p_max")
    im2 = axes[2].imshow(diff.T, origin="lower", cmap="magma")
    axes[2].set_title(f"|diff| max={float(diff.max()):.3g}")
    for ax in axes:
        ax.set_xlabel("z index")
        ax.set_ylabel("x index")
    fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(figure_path, dpi=120)
    plt.close(fig)


def build_report_lines(result: dict[str, object]) -> list[str]:
    metrics = result["metrics"]
    source_mask = metrics["source_mask"]
    source_weighted = metrics["source_weighted_mask"]
    p_max = metrics["p_max"]

    lines = [
        "at_linear_array_transducer parity report",
        "=======================================",
        "",
        "source mask overlap",
        "-------------------",
        f"  kwave_active_cells: {source_mask['kwave_active_cells']}",
        f"  pykwavers_active_cells: {source_mask['pykwavers_active_cells']}",
        f"  active_cell_ratio: {source_mask['active_cell_ratio']}",
        f"  intersection: {source_mask['intersection']}",
        f"  iou: {source_mask['iou']}",
        f"  dice: {source_mask['dice']}",
        "",
        "source weighted-mask parity",
        "---------------------------",
        f"  source_mode: {SOURCE_MODE}",
        f"  compatibility_mode: {COMPATIBILITY_MODE}",
    ]
    for key, value in source_weighted.items():
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            "p_max field parity",
            "------------------",
        ]
    )
    for key, value in p_max.items():
        lines.append(f"  {key}: {value}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached results and recompute both runs")
    parser.add_argument("--allow-failure", action="store_true", help="Do not exit non-zero when parity targets fail")
    args = parser.parse_args()
    global REFRESH_CACHE
    if args.no_cache:
        REFRESH_CACHE = True

    result = run_comparison()

    metrics = result["metrics"]
    source_mask = metrics["source_mask"]
    source_weighted = metrics["source_weighted_mask"]
    p_max = metrics["p_max"]

    source_mask_ok = source_mask["iou"] > 0.85 and source_mask["active_cell_ratio"] > 0.95
    source_weighted_ok = source_weighted["pearson_r"] > 0.20 and source_weighted["psnr_db"] > 25.0
    p_max_ok = p_max["pearson_r"] > 0.95 and abs(p_max["rms_ratio"] - 1.0) < 0.10 and p_max["psnr_db"] > 25.0

    status = "PASS" if (source_mask_ok and source_weighted_ok and p_max_ok) else "FAIL"

    report_lines = build_report_lines(result)
    report_lines.append(f"parity_status: {status}")
    save_comparison_figure(result, FIGURE_PATH)
    save_text_report(METRICS_PATH, "at_linear_array_transducer parity report", report_lines)

    print(f"[at_linear_array_transducer_compare] => {status}")
    print(f"  source mask: iou={source_mask['iou']:.4f} ratio={source_mask['active_cell_ratio']:.4f}")
    print(f"  source weighted mask: pearson_r={source_weighted['pearson_r']:.4f} psnr={source_weighted['psnr_db']:.2f}")
    print(f"  p_max: pearson_r={p_max['pearson_r']:.4f} rms_ratio={p_max['rms_ratio']:.4f} psnr={p_max['psnr_db']:.2f}")

    if status == "FAIL" and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
