#!/usr/bin/env python3
r"""
sd_directivity_modelling_2D_compare.py
======================================
Side-by-side comparison of k-wave-python vs pykwavers for the vendored
``sd_directivity_modelling_2D`` example.

Mathematical specification
--------------------------
For each source position ``i`` on the semicircle, let

``s_i(t) = sum_j p_{i,j}(t)``

be the detector-integrated trace across the line sensor points ``j``. The
directivity profile is then

``d_i = max_{t in W} s_i(t)``,

where ``W`` is the late-time window used by the vendored example. The parity
contract compares the full trace matrix ``S = [s_i]`` and the derived
directivity curve ``d`` between k-wave-python and pykwavers after applying the
one-sample recorder alignment used throughout the comparison scripts.

The script writes:

* ``output/sd_directivity_modelling_2D_compare.png``
* ``output/sd_directivity_modelling_2D_directivity.png``
* ``output/sd_directivity_modelling_2D_metrics.txt``
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    compute_trace_metrics,
    normalize_sensor_matrix,
    save_text_report,
)

_ROOT = bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.conversion import cart2grid
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matlab import matlab_find


GRID_SIZE_POINTS = Vector([128, 128])
GRID_SIZE_METERS = Vector([50e-3, 50e-3])
GRID_SPACING_METERS = GRID_SIZE_METERS / GRID_SIZE_POINTS
NX = int(GRID_SIZE_POINTS.x)
NY = int(GRID_SIZE_POINTS.y)
DX = float(GRID_SPACING_METERS.x)
DY = float(GRID_SPACING_METERS.y)

SOUND_SPEED = 1500.0
DENSITY = 1000.0
NT = 350
DT = 7.0e-8
PML_SIZE = 20

SENSOR_WIDTH = 20
SOURCE_RADIUS = 30
SOURCE_POINTS = 11
SOURCE_F0 = 0.25e6
SOURCE_MAG = 1.0

TRACE_WINDOW_START = 199
TRACE_WINDOW_STOP = 349

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "sd_directivity_modelling_2D_compare.png"
DIRECTIVITY_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "sd_directivity_modelling_2D_directivity.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "sd_directivity_modelling_2D_metrics.txt"

KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "sd_directivity_modelling_2D_kwave_cache.npz"
PYKWAVERS_CACHE = DEFAULT_OUTPUT_DIR / "sd_directivity_modelling_2D_pykwavers_cache.npz"

PARITY_THRESHOLDS = {
    "matrix": {
        "pearson_r": 0.99,
        "psnr_db": 30.0,
    },
    "trace": {
        "pearson_r_min": 0.99,
        "rms_ratio_min": 0.99,
        "rms_ratio_max": 1.01,
    },
    "directivity": {
        "pearson_r": 0.99,
        "rms_ratio_min": 0.99,
        "rms_ratio_max": 1.01,
        "peak_ratio_min": 0.99,
        "peak_ratio_max": 1.01,
    },
}

CACHE_VERSION = 1
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"


def _source_angle_rad(kgrid: kWaveGrid, source_position: int) -> float:
    """Return the polar angle used by the vendored example for one source."""
    x, y = np.unravel_index(int(source_position) - 1, kgrid.y.shape, order="F")
    return float(np.arctan2(kgrid.y[x, y], kgrid.x[x, y]))


def _build_example_inputs() -> dict[str, object]:
    """Construct the shared k-Wave / pykwavers configuration for one run."""
    kgrid = kWaveGrid(GRID_SIZE_POINTS, GRID_SPACING_METERS)
    kgrid.setTime(NT, DT)
    medium = kWaveMedium(sound_speed=SOUND_SPEED, density=DENSITY)

    sensor_mask = np.zeros((NX, NY), dtype=bool)
    sensor_mask[
        NX // 2,
        (NY // 2 - SENSOR_WIDTH // 2) : (NY // 2 + SENSOR_WIDTH // 2) + 1,
    ] = True
    sensor_count = int(np.count_nonzero(sensor_mask))

    source_circle = make_cart_circle(
        SOURCE_RADIUS * DX,
        SOURCE_POINTS,
        Vector([0, 0]),
        np.pi,
    )
    source_circle, _, _ = cart2grid(kgrid, source_circle)
    source_positions = np.asarray(matlab_find(source_circle, val=1, mode="eq"), dtype=np.int64).ravel()
    if source_positions.size != SOURCE_POINTS:
        raise AssertionError(
            f"Expected {SOURCE_POINTS} source positions, got {source_positions.size}"
        )

    source_angles = np.asarray(
        [_source_angle_rad(kgrid, int(source_position)) for source_position in source_positions],
        dtype=np.float64,
    )

    source_signal = SOURCE_MAG * np.sin(2.0 * np.pi * SOURCE_F0 * np.asarray(kgrid.t_array, dtype=np.float64).ravel())
    source_signal = np.asarray(filter_time_series(kgrid, medium, source_signal.reshape(1, -1)), dtype=np.float64).ravel()
    if source_signal.size != NT:
        raise AssertionError(f"Expected source signal length {NT}, got {source_signal.size}")

    return {
        "kgrid": kgrid,
        "medium": medium,
        "sensor_mask": sensor_mask,
        "sensor_count": sensor_count,
        "source_positions": source_positions,
        "source_angles": source_angles,
        "source_signal": source_signal,
    }


def _load_cached_result(path: Path) -> dict[str, np.ndarray | float] | None:
    if REFRESH_CACHE or not path.exists():
        return None
    cached = np.load(path, allow_pickle=False)
    version = int(np.asarray(cached["cache_version"]).reshape(())) if "cache_version" in cached.files else 0
    if version != CACHE_VERSION:
        return None
    return {
        "traces": np.asarray(cached["traces"], dtype=np.float64),
        "time": np.asarray(cached["time"], dtype=np.float64),
        "source_positions": np.asarray(cached["source_positions"], dtype=np.int64),
        "source_angles": np.asarray(cached["source_angles"], dtype=np.float64),
        "runtime_s": float(cached["runtime_s"]),
        "sensor_count": float(cached["sensor_count"]),
    }


def _sum_detector_trace(pressure: np.ndarray, sensor_count: int) -> np.ndarray:
    pressure_matrix = normalize_sensor_matrix(pressure, expected_sensors=sensor_count)
    return np.sum(pressure_matrix, axis=0, dtype=np.float64)


def _source_mask_2d_for_position(source_position: int) -> np.ndarray:
    source_mask = np.zeros((NX, NY), dtype=np.float64)
    source_mask[np.unravel_index(int(source_position) - 1, source_mask.shape, order="F")] = 1.0
    return source_mask


def _directivity_window(n_time_samples: int) -> slice:
    """Return the late-time window used for the directivity peak metric."""
    start = min(TRACE_WINDOW_START, max(0, n_time_samples - 1))
    stop = min(TRACE_WINDOW_STOP, n_time_samples)
    if stop <= start:
        stop = min(n_time_samples, start + 1)
    return slice(start, stop)


def _align_traces(
    kw_trace: np.ndarray,
    py_trace: np.ndarray,
    kw_time: np.ndarray,
    py_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align the recorder phase offset between k-Wave and pykwavers."""
    kw_trace = np.asarray(kw_trace, dtype=np.float64).ravel()
    py_trace = np.asarray(py_trace, dtype=np.float64).ravel()
    kw_time = np.asarray(kw_time, dtype=np.float64).ravel()
    py_time = np.asarray(py_time, dtype=np.float64).ravel()

    if kw_trace.size > 1 and py_trace.size > 1:
        kw_trace = kw_trace[1:]
        py_trace = py_trace[:-1]
        kw_time = kw_time[1:]
        py_time = py_time[:-1]

    n = min(kw_trace.size, py_trace.size, kw_time.size, py_time.size)
    return kw_trace[:n], py_trace[:n], kw_time[:n]


def run_kwave_reference() -> dict[str, np.ndarray | float]:
    """Run the k-wave-python reference for all source angles."""
    cached = _load_cached_result(KWAVE_CACHE)
    if cached is not None:
        return cached

    inputs = _build_example_inputs()
    kgrid = inputs["kgrid"]
    medium = inputs["medium"]
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)
    sensor_count = int(inputs["sensor_count"])
    source_positions = np.asarray(inputs["source_positions"], dtype=np.int64)
    source_angles = np.asarray(inputs["source_angles"], dtype=np.float64)
    source_signal = np.asarray(inputs["source_signal"], dtype=np.float64)

    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]

    traces = np.zeros((source_positions.size, NT), dtype=np.float64)
    total_runtime_s = 0.0

    for idx, source_position in enumerate(source_positions):
        source = kSource()
        source.p_mask = _source_mask_2d_for_position(int(source_position))
        source.p = source_signal.reshape(1, -1)

        sim_options = SimulationOptions(
            pml_inside=True,
            pml_size=PML_SIZE,
            data_cast="single",
            save_to_disk=True,
            input_filename=f"sd_directivity_modelling_2D_kwave_{idx}.h5",
            data_path=tempfile.gettempdir(),
        )
        exec_options = SimulationExecutionOptions(
            is_gpu_simulation=False,
            verbose_level=0,
            show_sim_log=False,
        )

        start = time.perf_counter()
        sensor_data = kspaceFirstOrder2DC(
            medium=medium,
            kgrid=kgrid,
            source=source,
            sensor=sensor,
            simulation_options=sim_options,
            execution_options=exec_options,
        )
        total_runtime_s += time.perf_counter() - start

        pressure = normalize_sensor_matrix(sensor_data["p"], expected_sensors=sensor_count)
        traces[idx] = _sum_detector_trace(pressure, sensor_count)

    output = {
        "traces": traces,
        "time": np.asarray(kgrid.t_array, dtype=np.float64).ravel(),
        "source_positions": source_positions,
        "source_angles": source_angles,
        "runtime_s": total_runtime_s,
        "sensor_count": float(sensor_count),
    }
    np.savez(KWAVE_CACHE, cache_version=CACHE_VERSION, **output)
    return output


def run_pykwavers_reference() -> dict[str, np.ndarray | float]:
    """Run the pykwavers counterpart for all source angles."""
    cached = _load_cached_result(PYKWAVERS_CACHE)
    if cached is not None:
        return cached

    inputs = _build_example_inputs()
    kgrid = inputs["kgrid"]
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)
    sensor_count = int(inputs["sensor_count"])
    source_positions = np.asarray(inputs["source_positions"], dtype=np.int64)
    source_angles = np.asarray(inputs["source_angles"], dtype=np.float64)
    source_signal = np.asarray(inputs["source_signal"], dtype=np.float64)

    grid = pkw.Grid(nx=NX, ny=NY, nz=1, dx=DX, dy=DY, dz=DX)
    medium = pkw.Medium.homogeneous(sound_speed=SOUND_SPEED, density=DENSITY)
    sensor = pkw.Sensor.from_mask(sensor_mask[:, :, None])

    traces = np.zeros((source_positions.size, NT), dtype=np.float64)
    total_runtime_s = 0.0

    for idx, source_position in enumerate(source_positions):
        source_mask_3d = _source_mask_2d_for_position(int(source_position))[:, :, None]
        source = pkw.Source.from_mask(source_mask_3d, source_signal, SOURCE_F0, mode="additive")

        sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
        sim.set_pml_size(PML_SIZE)
        sim.set_pml_inside(True)

        start = time.perf_counter()
        result = sim.run(time_steps=int(kgrid.Nt), dt=DT)
        total_runtime_s += time.perf_counter() - start

        pressure = normalize_sensor_matrix(result.sensor_data, expected_sensors=sensor_count)
        traces[idx] = _sum_detector_trace(pressure, sensor_count)

    output = {
        "traces": traces,
        "time": np.asarray(result.time, dtype=np.float64).ravel(),
        "source_positions": source_positions,
        "source_angles": source_angles,
        "runtime_s": total_runtime_s,
        "sensor_count": float(sensor_count),
    }
    np.savez(PYKWAVERS_CACHE, cache_version=CACHE_VERSION, **output)
    return output


def run_comparison() -> dict[str, object]:
    """Run the full parity comparison and return aligned matrices plus metrics."""
    kw = run_kwave_reference()
    py = run_pykwavers_reference()

    kw_traces = np.asarray(kw["traces"], dtype=np.float64)
    py_traces = np.asarray(py["traces"], dtype=np.float64)
    if kw_traces.shape != py_traces.shape:
        raise AssertionError(f"Trace matrix mismatch: {kw_traces.shape} != {py_traces.shape}")

    kw_time = np.asarray(kw["time"], dtype=np.float64)
    py_time = np.asarray(py["time"], dtype=np.float64)
    source_angles_deg = np.degrees(np.asarray(kw["source_angles"], dtype=np.float64))

    aligned_kw_rows: list[np.ndarray] = []
    aligned_py_rows: list[np.ndarray] = []
    trace_metrics: dict[str, dict[str, float]] = {}
    aligned_time: np.ndarray | None = None

    for idx in range(kw_traces.shape[0]):
        kw_trace, py_trace, aligned_time_row = _align_traces(
            kw_traces[idx],
            py_traces[idx],
            kw_time,
            py_time,
        )
        if aligned_time is None:
            aligned_time = aligned_time_row
        elif aligned_time.shape != aligned_time_row.shape or not np.allclose(aligned_time, aligned_time_row):
            raise AssertionError("Aligned time vectors differ across source positions")

        aligned_kw_rows.append(kw_trace)
        aligned_py_rows.append(py_trace)
        trace_metrics[f"source_{idx:02d}"] = compute_trace_metrics(kw_trace, py_trace)

    aligned_kw = np.vstack(aligned_kw_rows)
    aligned_py = np.vstack(aligned_py_rows)
    matrix_metrics = compute_image_metrics(aligned_kw, aligned_py)

    trace_pearson = np.asarray([m["pearson_r"] for m in trace_metrics.values()], dtype=np.float64)
    trace_rms = np.asarray([m["rms_ratio"] for m in trace_metrics.values()], dtype=np.float64)
    trace_rmse = np.asarray([m["rmse"] for m in trace_metrics.values()], dtype=np.float64)
    trace_peak_ratio = np.asarray([m["peak_ratio"] for m in trace_metrics.values()], dtype=np.float64)

    trace_summary = {
        "pearson_r_mean": float(np.mean(trace_pearson)),
        "pearson_r_min": float(np.min(trace_pearson)),
        "pearson_r_median": float(np.median(trace_pearson)),
        "rms_ratio_mean": float(np.mean(trace_rms)),
        "rms_ratio_median": float(np.median(trace_rms)),
        "rmse_mean": float(np.mean(trace_rmse)),
        "rmse_median": float(np.median(trace_rmse)),
        "peak_ratio_mean": float(np.mean(trace_peak_ratio)),
        "peak_ratio_median": float(np.median(trace_peak_ratio)),
    }

    sort_order = np.argsort(source_angles_deg)
    source_angles_deg_sorted = source_angles_deg[sort_order]
    aligned_kw_sorted = aligned_kw[sort_order]
    aligned_py_sorted = aligned_py[sort_order]

    directivity_window = _directivity_window(aligned_kw_sorted.shape[1])
    kw_directivity = np.max(aligned_kw_sorted[:, directivity_window], axis=1)
    py_directivity = np.max(aligned_py_sorted[:, directivity_window], axis=1)
    directivity_metrics = compute_trace_metrics(kw_directivity, py_directivity)

    return {
        "kwave": kw,
        "pykwavers": py,
        "aligned": {
            "time": aligned_time,
            "source_angles_deg": source_angles_deg_sorted,
            "kw_traces": aligned_kw_sorted,
            "py_traces": aligned_py_sorted,
            "kw_directivity": kw_directivity,
            "py_directivity": py_directivity,
            "sort_order": sort_order,
        },
        "matrix_metrics": matrix_metrics,
        "trace_metrics": trace_metrics,
        "trace_summary": trace_summary,
        "directivity_metrics": directivity_metrics,
    }


def plot_comparison(result: dict[str, object]) -> None:
    """Plot the aligned trace matrices for both engines."""
    aligned = result["aligned"]  # type: ignore[assignment]
    kw_traces = np.asarray(aligned["kw_traces"], dtype=np.float64)  # type: ignore[index]
    py_traces = np.asarray(aligned["py_traces"], dtype=np.float64)  # type: ignore[index]
    time_us = np.asarray(aligned["time"], dtype=np.float64) * 1e6  # type: ignore[index]
    source_angles_deg = np.asarray(aligned["source_angles_deg"], dtype=np.float64)  # type: ignore[index]

    vmax = float(max(np.max(np.abs(kw_traces)), np.max(np.abs(py_traces))))
    if vmax <= 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, constrained_layout=True)
    panels = [
        (axes[0], kw_traces, "k-wave-python"),
        (axes[1], py_traces, "pykwavers"),
    ]
    for ax, matrix, title in panels:
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Time [us]")
        ax.set_ylabel("Source angle [deg]")
        xticks = np.linspace(0, matrix.shape[1] - 1, 5, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{time_us[idx]:.1f}" for idx in xticks])
        ax.set_yticks(np.arange(source_angles_deg.size))
        ax.set_yticklabels([f"{angle:.1f}" for angle in source_angles_deg])
        ax.grid(False)

    fig.colorbar(im, ax=axes, location="right", shrink=0.9, label="Summed detector pressure [Pa]")
    fig.suptitle("sd_directivity_modelling_2D: aligned detector trace matrix")
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_directivity(result: dict[str, object]) -> None:
    """Plot the directivity curve derived from the late-time trace window."""
    aligned = result["aligned"]  # type: ignore[assignment]
    source_angles_deg = np.asarray(aligned["source_angles_deg"], dtype=np.float64)  # type: ignore[index]
    kw_directivity = np.asarray(aligned["kw_directivity"], dtype=np.float64)  # type: ignore[index]
    py_directivity = np.asarray(aligned["py_directivity"], dtype=np.float64)  # type: ignore[index]
    directivity = result["directivity_metrics"]  # type: ignore[assignment]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(source_angles_deg, kw_directivity, color="black", marker="o", linewidth=1.5, label="k-wave-python")
    ax.plot(source_angles_deg, py_directivity, color="crimson", marker="o", linewidth=1.2, label="pykwavers")
    ax.set_xlabel("Source angle [deg]")
    ax.set_ylabel("Peak detector pressure [Pa]")
    ax.set_title(
        "sd_directivity_modelling_2D directivity\n"
        f"Pearson r={directivity['pearson_r']:.6f}, "
        f"RMS ratio={directivity['rms_ratio']:.6f}, "
        f"peak ratio={directivity['peak_ratio']:.6f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(DIRECTIVITY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Execute the comparison and print the metrics."""
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for sd_directivity_modelling_2D.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Delete cached results and force a fresh run.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Exit 0 even when parity targets fail.",
    )
    args = parser.parse_args()

    if args.no_cache:
        for cache_path in (KWAVE_CACHE, PYKWAVERS_CACHE):
            if cache_path.exists():
                cache_path.unlink()

    result = run_comparison()
    matrix_metrics = result["matrix_metrics"]  # type: ignore[assignment]
    trace_metrics = result["trace_metrics"]  # type: ignore[assignment]
    trace_summary = result["trace_summary"]  # type: ignore[assignment]
    directivity = result["directivity_metrics"]  # type: ignore[assignment]
    kw = result["kwave"]  # type: ignore[assignment]
    py = result["pykwavers"]  # type: ignore[assignment]

    print("=" * 80)
    print("sd_directivity_modelling_2D: k-wave-python vs pykwavers")
    print("=" * 80)
    print(f"k-wave runtime [s]     = {kw['runtime_s']:.6f}")
    print(f"pykwavers runtime [s]  = {py['runtime_s']:.6f}")
    print(f"Matrix Pearson r       = {matrix_metrics['pearson_r']:.6f}")
    print(f"Matrix RMS ratio       = {matrix_metrics['rms_ratio']:.6f}")
    print(f"Matrix PSNR [dB]       = {matrix_metrics['psnr_db']:.6f}")
    print(f"Trace Pearson r mean   = {trace_summary['pearson_r_mean']:.6f}")
    print(f"Trace Pearson r min    = {trace_summary['pearson_r_min']:.6f}")
    print(f"Directivity Pearson r  = {directivity['pearson_r']:.6f}")
    print(f"Directivity RMS ratio  = {directivity['rms_ratio']:.6f}")
    print(f"Directivity peak ratio = {directivity['peak_ratio']:.6f}")

    plot_comparison(result)
    plot_directivity(result)

    report_lines = [
        "sd_directivity_modelling_2D parity metrics",
        f"kwave_runtime_s: {kw['runtime_s']:.8f}",
        f"pykwavers_runtime_s: {py['runtime_s']:.8f}",
        f"matrix_pearson_r: {matrix_metrics['pearson_r']:.8f}",
        f"matrix_rms_ratio: {matrix_metrics['rms_ratio']:.8f}",
        f"matrix_psnr_db: {matrix_metrics['psnr_db']:.8f}",
        f"trace_pearson_r_mean: {trace_summary['pearson_r_mean']:.8f}",
        f"trace_pearson_r_min: {trace_summary['pearson_r_min']:.8f}",
        f"trace_rms_ratio_mean: {trace_summary['rms_ratio_mean']:.8f}",
        f"trace_rmse_median: {trace_summary['rmse_median']:.8e}",
        f"trace_peak_ratio_mean: {trace_summary['peak_ratio_mean']:.8f}",
        f"directivity_pearson_r: {directivity['pearson_r']:.8f}",
        f"directivity_rms_ratio: {directivity['rms_ratio']:.8f}",
        f"directivity_rmse: {directivity['rmse']:.8e}",
        f"directivity_peak_ratio: {directivity['peak_ratio']:.8f}",
        "",
    ]

    source_angles_deg = np.degrees(np.asarray(result["kwave"]["source_angles"], dtype=np.float64))  # type: ignore[index]
    for idx, (source_key, metrics) in enumerate(trace_metrics.items()):
        report_lines.extend(
            [
                f"{source_key}:",
                f"  source_angle_deg: {source_angles_deg[idx]:.6f}",
                f"  pearson_r: {metrics['pearson_r']:.8f}",
                f"  rms_ratio: {metrics['rms_ratio']:.8f}",
                f"  rmse: {metrics['rmse']:.8e}",
                f"  max_abs_diff: {metrics['max_abs_diff']:.8e}",
                f"  peak_ratio: {metrics['peak_ratio']:.8f}",
                "",
            ]
        )

    matrix_thresholds = PARITY_THRESHOLDS["matrix"]
    trace_thresholds = PARITY_THRESHOLDS["trace"]
    directivity_thresholds = PARITY_THRESHOLDS["directivity"]
    trace_ok = (
        matrix_metrics["pearson_r"] > matrix_thresholds["pearson_r"]
        and matrix_metrics["psnr_db"] > matrix_thresholds["psnr_db"]
        and trace_summary["pearson_r_min"] > trace_thresholds["pearson_r_min"]
        and trace_thresholds["rms_ratio_min"]
        <= trace_summary["rms_ratio_mean"]
        <= trace_thresholds["rms_ratio_max"]
    )
    directivity_ok = (
        directivity["pearson_r"] > directivity_thresholds["pearson_r"]
        and directivity_thresholds["rms_ratio_min"]
        <= directivity["rms_ratio"]
        <= directivity_thresholds["rms_ratio_max"]
        and directivity_thresholds["peak_ratio_min"]
        <= directivity["peak_ratio"]
        <= directivity_thresholds["peak_ratio_max"]
    )
    overall_status = "PASS" if (trace_ok and directivity_ok) else "FAIL"
    report_lines.append(f"parity_status: {overall_status}")
    save_text_report(METRICS_PATH, "sd_directivity_modelling_2D parity metrics", report_lines)
    print(f"Status: {overall_status}")

    return 0 if overall_status == "PASS" or args.allow_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
