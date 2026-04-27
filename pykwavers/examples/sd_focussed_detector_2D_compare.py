#!/usr/bin/env python3
r"""
sd_focussed_detector_2D_compare.py
===================================
Side-by-side comparison of k-wave-python vs pykwavers for the 2D focussed
detector example.

This script mirrors the vendored k-wave-python example in
``external/k-wave-python/examples/sd_focussed_detector_2D/sd_focussed_detector_2D.py``.
Two disc sources are simulated against a semicircular detector:

* Source 1: on-axis, placed at the geometric focus of the detector.
* Source 2: off-axis, shifted horizontally by 20 grid points.

The detector output is the spatial mean over all active detector points,
which is the quantity plotted in the original example:

.. math::

   \bar p(t) = \frac{1}{N_s} \sum_{i=1}^{N_s} p_i(t).

The comparison reports:

* trace parity for each source position,
* directivity energy contrast between the on-axis and off-axis cases,
* runtime and cached artifacts for repeated runs.
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
    compute_trace_metrics,
    expand_pml_outside_shape,
    normalize_sensor_matrix,
    pad_volume_for_pml_outside,
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
from kwave.utils.mapgen import make_circle, make_disc


GRID_SIZE = Vector([180, 180])
GRID_SPACING = Vector([0.1e-3, 0.1e-3])
SOUND_SPEED = 1500.0
DENSITY = 1000.0
PML_SIZE = (10, 10)
T_END = 11e-6
SOURCE_RADIUS = 4
SENSOR_RADIUS = 65
ARC_ANGLE = np.pi
SOURCE_OFFSETS = {"on_axis": 0, "off_axis": 20}
SOURCE_LABELS = {
    "on_axis": "Source on focus",
    "off_axis": "Source off focus",
}

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "sd_focussed_detector_2D_compare.png"
DIRECTIVITY_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "sd_focussed_detector_2D_directivity.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "sd_focussed_detector_2D_metrics.txt"

KWAVE_CACHE = {
    "on_axis": DEFAULT_OUTPUT_DIR / "sd_focussed_detector_2D_kwave_on_axis.npz",
    "off_axis": DEFAULT_OUTPUT_DIR / "sd_focussed_detector_2D_kwave_off_axis.npz",
}
PYKWAVERS_CACHE = {
    "on_axis": DEFAULT_OUTPUT_DIR / "sd_focussed_detector_2D_pykwavers_on_axis.npz",
    "off_axis": DEFAULT_OUTPUT_DIR / "sd_focussed_detector_2D_pykwavers_off_axis.npz",
}

CACHE_VERSION = 1
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"


def _build_example_inputs(source_offset_y: int) -> dict[str, object]:
    """Construct the shared k-Wave / pykwavers configuration for one source."""
    kgrid = kWaveGrid(GRID_SIZE, GRID_SPACING)
    medium = kWaveMedium(sound_speed=SOUND_SPEED, density=DENSITY, alpha_coeff=0.0, alpha_power=1.5)
    kgrid.makeTime(medium.sound_speed, t_end=T_END)

    sensor_mask = np.asarray(
        make_circle(GRID_SIZE, GRID_SIZE / 2 + 1, SENSOR_RADIUS, ARC_ANGLE),
        dtype=bool,
    )
    sensor_count = int(np.count_nonzero(sensor_mask))

    source_center = GRID_SIZE / 2 + Vector([0, source_offset_y])
    source_p0 = 2.0 * np.asarray(make_disc(GRID_SIZE, source_center, SOURCE_RADIUS), dtype=np.float64)

    return {
        "kgrid": kgrid,
        "medium": medium,
        "sensor_mask": sensor_mask,
        "sensor_count": sensor_count,
        "source_p0": source_p0,
        "dt": float(kgrid.dt),
        "Nt": int(kgrid.Nt),
    }


def _load_cached_result(path: Path) -> dict[str, np.ndarray | float] | None:
    if REFRESH_CACHE or not path.exists():
        return None
    cached = np.load(path, allow_pickle=False)
    version = int(np.asarray(cached["cache_version"]).reshape(())) if "cache_version" in cached.files else 0
    if version != CACHE_VERSION:
        return None
    return {
        "pressure": np.asarray(cached["pressure"], dtype=np.float64),
        "trace": np.asarray(cached["trace"], dtype=np.float64),
        "time": np.asarray(cached["time"], dtype=np.float64),
        "dt": float(cached["dt"]),
        "runtime_s": float(cached["runtime_s"]),
        "sensor_count": float(cached["sensor_count"]),
    }


def _mean_detector_trace(pressure: np.ndarray, sensor_count: int) -> np.ndarray:
    if sensor_count <= 0:
        raise ValueError("sensor_count must be positive")
    pressure_matrix = normalize_sensor_matrix(pressure, expected_sensors=sensor_count)
    trace = np.sum(pressure_matrix, axis=0) / float(sensor_count)
    return np.asarray(trace, dtype=np.float64)


def _align_traces(
    kw_trace: np.ndarray,
    py_trace: np.ndarray,
    kw_time: np.ndarray,
    py_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align k-Wave and pykwavers traces to the shared detector sampling convention."""
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
    return kw_trace[:n], py_trace[:n], kw_time[:n], py_time[:n]


def run_kwave_reference(source_tag: str, source_offset_y: int) -> dict[str, np.ndarray | float]:
    """Run the k-wave-python reference simulation for one source position."""
    cache_path = KWAVE_CACHE[source_tag]
    cached = _load_cached_result(cache_path)
    if cached is not None:
        return cached

    inputs = _build_example_inputs(source_offset_y)
    kgrid = inputs["kgrid"]
    medium = inputs["medium"]
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)
    source_p0 = np.asarray(inputs["source_p0"], dtype=np.float64)
    sensor_count = int(inputs["sensor_count"])

    source = kSource()
    source.p0 = source_p0

    sensor = kSensor(sensor_mask.astype(np.int32))
    sensor.record = ["p"]

    input_filename = f"sd_focussed_detector_2D_{source_tag}.h5"
    simulation_options = SimulationOptions(
        pml_inside=True,
        pml_size=Vector(list(PML_SIZE)),
        smooth_p0=False,
        save_to_disk=True,
        input_filename=input_filename,
        data_path=tempfile.gettempdir(),
    )
    execution_options = SimulationExecutionOptions(
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
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    runtime_s = time.perf_counter() - start

    pressure = normalize_sensor_matrix(sensor_data["p"], expected_sensors=sensor_count)
    trace = _mean_detector_trace(pressure, sensor_count)
    result = {
        "pressure": pressure,
        "trace": trace,
        "time": np.asarray(kgrid.t_array, dtype=np.float64).ravel(),
        "dt": float(kgrid.dt),
        "runtime_s": runtime_s,
        "sensor_count": float(sensor_count),
    }
    np.savez(cache_path, cache_version=CACHE_VERSION, **result)
    return result


def run_pykwavers_reference(source_tag: str, source_offset_y: int) -> dict[str, np.ndarray | float]:
    """Run the pykwavers simulation for one source position."""
    cache_path = PYKWAVERS_CACHE[source_tag]
    cached = _load_cached_result(cache_path)
    if cached is not None:
        return cached

    inputs = _build_example_inputs(source_offset_y)
    kgrid = inputs["kgrid"]
    medium = inputs["medium"]
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)
    source_p0 = np.asarray(inputs["source_p0"], dtype=np.float64)
    sensor_count = int(inputs["sensor_count"])

    expanded_grid_points = expand_pml_outside_shape(GRID_SIZE, PML_SIZE)
    expanded_source_p0 = pad_volume_for_pml_outside(source_p0, PML_SIZE)
    expanded_sensor_mask = pad_volume_for_pml_outside(sensor_mask, PML_SIZE).astype(bool)

    grid = pkw.Grid(
        expanded_grid_points[0],
        expanded_grid_points[1],
        expanded_grid_points[2],
        float(GRID_SPACING[0]),
        float(GRID_SPACING[1]),
        float(GRID_SPACING[0]),
    )
    medium_pk = pkw.Medium.homogeneous(sound_speed=SOUND_SPEED, density=DENSITY)
    source_pk = pkw.Source.from_initial_pressure(expanded_source_p0)
    sensor_pk = pkw.Sensor.from_mask(expanded_sensor_mask)

    sim = pkw.Simulation(grid, medium_pk, source_pk, sensor_pk, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE[0])
    sim.set_pml_inside(True)

    start = time.perf_counter()
    result = sim.run(time_steps=int(kgrid.Nt), dt=float(kgrid.dt))
    runtime_s = time.perf_counter() - start

    pressure = normalize_sensor_matrix(result.sensor_data, expected_sensors=sensor_count)
    trace = _mean_detector_trace(pressure, sensor_count)
    output = {
        "pressure": pressure,
        "trace": trace,
        "time": np.asarray(result.time, dtype=np.float64).ravel(),
        "dt": float(result.dt),
        "runtime_s": runtime_s,
        "sensor_count": float(sensor_count),
    }
    np.savez(cache_path, cache_version=CACHE_VERSION, **output)
    return output


def run_comparison() -> dict[str, object]:
    """Run the on-axis and off-axis source comparisons."""
    kwave_results: dict[str, dict[str, np.ndarray | float]] = {}
    pykwavers_results: dict[str, dict[str, np.ndarray | float]] = {}
    trace_metrics: dict[str, dict[str, float]] = {}

    for tag, source_offset_y in SOURCE_OFFSETS.items():
        kw = run_kwave_reference(tag, source_offset_y)
        py = run_pykwavers_reference(tag, source_offset_y)

        kw_trace, py_trace, kw_time, py_time = _align_traces(
            kw["trace"], py["trace"], kw["time"], py["time"]
        )

        kwave_results[tag] = {**kw, "trace": kw_trace, "time": kw_time}
        pykwavers_results[tag] = {**py, "trace": py_trace, "time": py_time}
        trace_metrics[tag] = compute_trace_metrics(kw_trace, py_trace)

    kw_on = np.asarray(kwave_results["on_axis"]["trace"], dtype=np.float64)
    kw_off = np.asarray(kwave_results["off_axis"]["trace"], dtype=np.float64)
    py_on = np.asarray(pykwavers_results["on_axis"]["trace"], dtype=np.float64)
    py_off = np.asarray(pykwavers_results["off_axis"]["trace"], dtype=np.float64)

    kw_directivity = float(np.sum(kw_on**2) / (np.sum(kw_off**2) + 1e-30))
    py_directivity = float(np.sum(py_on**2) / (np.sum(py_off**2) + 1e-30))
    directivity_metrics = {
        "kwave_ratio": kw_directivity,
        "pykwavers_ratio": py_directivity,
        "relative_error": float(abs(py_directivity - kw_directivity) / (abs(kw_directivity) + 1e-30)),
    }

    return {
        "kwave": kwave_results,
        "pykwavers": pykwavers_results,
        "trace_metrics": trace_metrics,
        "directivity_metrics": directivity_metrics,
    }


def plot_comparison(result: dict[str, object]) -> None:
    """Plot the on-axis and off-axis detector traces."""
    kwave = result["kwave"]  # type: ignore[assignment]
    pykwavers = result["pykwavers"]  # type: ignore[assignment]

    on_kw = kwave["on_axis"]  # type: ignore[index]
    off_kw = kwave["off_axis"]  # type: ignore[index]
    on_py = pykwavers["on_axis"]  # type: ignore[index]
    off_py = pykwavers["off_axis"]  # type: ignore[index]

    t_on = np.asarray(on_kw["time"], dtype=np.float64) * 1e6
    t_off = np.asarray(off_kw["time"], dtype=np.float64) * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    rows = [
        (axes[0], t_on, on_kw, on_py, "Source on focus"),
        (axes[1], t_off, off_kw, off_py, "Source off focus"),
    ]
    for ax, time_us, kw_res, py_res, label in rows:
        ax.plot(time_us, kw_res["trace"], color="black", linewidth=1.5, alpha=0.85, label="k-wave-python")
        ax.plot(time_us, py_res["trace"], color="crimson", linewidth=1.2, alpha=0.85, label="pykwavers")
        ax.set_title(
            f"{label}  |  k-wave E={np.sum(np.asarray(kw_res['trace'])**2):.4e}  "
            f"pykwavers E={np.sum(np.asarray(py_res['trace'])**2):.4e}"
        )
        ax.set_xlabel("Time [µs]")
        ax.set_ylabel("Average detector pressure [Pa]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "sd_focussed_detector_2D: k-wave-python vs pykwavers\n"
        "Semicircular detector average pressure traces",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_directivity(result: dict[str, object]) -> None:
    """Plot the directivity contrast for both engines on one panel."""
    kwave = result["kwave"]  # type: ignore[assignment]
    pykwavers = result["pykwavers"]  # type: ignore[assignment]
    directivity = result["directivity_metrics"]  # type: ignore[assignment]

    on_kw = kwave["on_axis"]  # type: ignore[index]
    off_kw = kwave["off_axis"]  # type: ignore[index]
    on_py = pykwavers["on_axis"]  # type: ignore[index]
    off_py = pykwavers["off_axis"]  # type: ignore[index]

    time_us = np.asarray(on_kw["time"], dtype=np.float64) * 1e6

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(time_us, on_kw["trace"], color="navy", linewidth=1.6, label="k-wave on-axis")
    ax.plot(time_us, off_kw["trace"], color="navy", linestyle="--", linewidth=1.2, label="k-wave off-axis")
    ax.plot(time_us, on_py["trace"], color="darkred", linewidth=1.4, label="pykwavers on-axis")
    ax.plot(time_us, off_py["trace"], color="darkred", linestyle="--", linewidth=1.0, label="pykwavers off-axis")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Average detector pressure [Pa]")
    ax.set_title(
        "Focused detector directivity\n"
        f"k-wave ratio={directivity['kwave_ratio']:.6f}, "
        f"pykwavers ratio={directivity['pykwavers_ratio']:.6f}, "
        f"relative error={directivity['relative_error']:.3e}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(DIRECTIVITY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Execute the comparison and print the metrics."""
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for sd_focussed_detector_2D."
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
        for cache_dict in (KWAVE_CACHE, PYKWAVERS_CACHE):
            for cache_path in cache_dict.values():
                if cache_path.exists():
                    cache_path.unlink()

    result = run_comparison()
    trace_metrics = result["trace_metrics"]  # type: ignore[assignment]
    directivity = result["directivity_metrics"]  # type: ignore[assignment]

    print("=" * 80)
    print("sd_focussed_detector_2D: k-wave-python vs pykwavers")
    print("=" * 80)
    for tag in ("on_axis", "off_axis"):
        metrics = trace_metrics[tag]  # type: ignore[index]
        print(f"[{tag}] Pearson r   = {metrics['pearson_r']:.6f}")
        print(f"[{tag}] RMS ratio   = {metrics['rms_ratio']:.6f}")
        print(f"[{tag}] RMSE        = {metrics['rmse']:.6e}")
        print(f"[{tag}] Peak ratio  = {metrics['peak_ratio']:.6f}")
    print(f"Directivity ratio (k-wave)     = {directivity['kwave_ratio']:.6f}")
    print(f"Directivity ratio (pykwavers)   = {directivity['pykwavers_ratio']:.6f}")
    print(f"Directivity relative error      = {directivity['relative_error']:.6e}")

    plot_comparison(result)
    plot_directivity(result)

    report_lines = [
        "sd_focussed_detector_2D parity metrics",
        f"kwave_directivity_ratio: {directivity['kwave_ratio']:.8f}",
        f"pykwavers_directivity_ratio: {directivity['pykwavers_ratio']:.8f}",
        f"directivity_relative_error: {directivity['relative_error']:.8e}",
        "",
    ]
    for tag in ("on_axis", "off_axis"):
        metrics = trace_metrics[tag]  # type: ignore[index]
        report_lines.extend(
            [
                f"{tag}:",
                f"  pearson_r: {metrics['pearson_r']:.8f}",
                f"  rms_ratio: {metrics['rms_ratio']:.8f}",
                f"  rmse: {metrics['rmse']:.8e}",
                f"  max_abs_diff: {metrics['max_abs_diff']:.8e}",
                f"  peak_ratio: {metrics['peak_ratio']:.8f}",
                "",
            ]
        )
    trace_ok = all(
        trace_metrics[tag]["pearson_r"] > 0.999
        and abs(trace_metrics[tag]["rms_ratio"] - 1.0) < 1e-2
        and trace_metrics[tag]["rmse"] < 1e-2
        and abs(trace_metrics[tag]["peak_ratio"] - 1.0) < 1e-2
        for tag in ("on_axis", "off_axis")
    )
    directivity_ok = (
        directivity["kwave_ratio"] > 1.0
        and directivity["pykwavers_ratio"] > 1.0
        and directivity["relative_error"] < 1e-2
    )
    overall_status = "PASS" if (trace_ok and directivity_ok) else "FAIL"
    report_lines.append(f"parity_status: {overall_status}")
    save_text_report(METRICS_PATH, "sd_focussed_detector_2D parity metrics", report_lines)
    print(f"Status: {overall_status}")

    return 0 if overall_status == "PASS" or args.allow_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
