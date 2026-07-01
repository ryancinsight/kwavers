#!/usr/bin/env python3
"""
Comparison: k-wave-python `at_array_as_sensor` vs pykwavers.

Mathematical specification
--------------------------
The vendored example defines a circular receiver array as a collection of arc
elements.  For each arc element with midpoint `a`, focus point `f`, curvature
radius `r`, and chord length `d`, the equivalent circle-center representation is

`u = (f - a) / ||f - a||`
`c = a + r u`
`phi = asin(d / (2r))`
`theta_mid = atan2(-u_y, -u_x)`
`theta in [theta_mid - phi, theta_mid + phi]`

This conversion is exact on a uniform Cartesian grid.  Translating the entire
coordinate frame by a constant vector preserves the discrete wave equation on a
uniform mesh, so the centered k-wave-python frame can be shifted into the
positive-origin pykwavers frame without changing the source/sensor geometry.

The comparison reports:

* binary mask parity for the array sensor geometry,
* weighted mask parity for the arc integration weights,
* raw detector-matrix parity,
* combined arc-trace parity after applying the same reduction rule as
  `kWaveArray.combine_sensor_data`.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from array_sensor_utils import (
    ArcElementGeometry,
    build_arc_element_geometry,
    combine_array_sensor_data,
)
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
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.mapgen import make_cart_circle, make_disc


GRID_POINTS = Vector([256, 256])
GRID_SPACING = Vector([0.5e-3, 0.5e-3])
NX = int(GRID_POINTS.x)
NY = int(GRID_POINTS.y)
DX = float(GRID_SPACING.x)
DY = float(GRID_SPACING.y)
DOMAIN_TRANSLATION = np.array([NX * DX / 2.0, NY * DY / 2.0], dtype=np.float64)

SOUND_SPEED = 1500.0
DENSITY = 1000.0
PML_SIZE = 20
ARC_RADIUS = 100e-3
ARC_DIAMETER = 8e-3
RING_RADIUS = 50e-3
NUM_ELEMENTS = 20

SOURCE_OFFSET = 20
SOURCE_RADIUS = 4

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "at_array_as_sensor_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "at_array_as_sensor_metrics.txt"

KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "at_array_as_sensor_kwave_cache.npz"
PYKWAVERS_CACHE = DEFAULT_OUTPUT_DIR / "at_array_as_sensor_pykwavers_cache.npz"
CACHE_VERSION = 1
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"


def _build_source_p0() -> np.ndarray:
    """Return the initial pressure field used by the vendored example."""
    source = kSource()
    source.p0 = make_disc(GRID_POINTS, Vector([GRID_POINTS.x / 4 + SOURCE_OFFSET, GRID_POINTS.y / 4]), SOURCE_RADIUS)
    source.p0[99:119, 59:199] = 1.0
    return np.asarray(source.p0, dtype=np.float64)


def _build_array_geometries() -> tuple[np.ndarray, list[ArcElementGeometry]]:
    """Return the centered k-wave arc positions and the translated pykwavers arcs."""
    focus_kw = np.array([0.0, 0.0], dtype=np.float64)
    arc_positions_kw = np.asarray(make_cart_circle(RING_RADIUS, NUM_ELEMENTS, Vector([0, 0])), dtype=np.float64)
    arc_geometries = [
        build_arc_element_geometry(
            arc_positions_kw[:, idx],
            ARC_RADIUS,
            ARC_DIAMETER,
            focus_kw,
            DOMAIN_TRANSLATION,
        )
        for idx in range(arc_positions_kw.shape[1])
    ]
    return arc_positions_kw, arc_geometries


def _build_kwave_array(arc_positions_kw: np.ndarray) -> KWaveArray_Kwave:
    """Build the reference k-wave-python `kWaveArray`."""
    array = KWaveArray_Kwave()
    focus_kw = Vector([0, 0])
    for idx in range(arc_positions_kw.shape[1]):
        arc_position = tuple(float(v) for v in arc_positions_kw[:, idx])
        array.add_arc_element(arc_position, ARC_RADIUS, ARC_DIAMETER, focus_kw)
    return array


def _build_pykwavers_array(arc_geometries: list[ArcElementGeometry]) -> pkw.KWaveArray:
    """Build the pykwavers `KWaveArray` with the translated arc geometry."""
    array = pkw.KWaveArray()
    for geom in arc_geometries:
        array.add_arc_element(
            geom.pykwavers_center,
            geom.radius_m,
            geom.diameter_m,
            geom.start_angle_deg,
            geom.end_angle_deg,
        )
    return array


def _build_pykwavers_single_element_masks(
    arc_geometries: list[ArcElementGeometry],
    grid: pkw.Grid,
) -> list[np.ndarray]:
    """Return one weighted mask per arc element, matching `combine_sensor_data`."""
    element_masks: list[np.ndarray] = []
    for geom in arc_geometries:
        array = pkw.KWaveArray()
        array.add_arc_element(
            geom.pykwavers_center,
            geom.radius_m,
            geom.diameter_m,
            geom.start_angle_deg,
            geom.end_angle_deg,
        )
        element_masks.append(np.squeeze(np.asarray(array.get_array_weighted_mask(grid), dtype=np.float64)))
    return element_masks


def _align_sensor_matrix(pressure: np.ndarray, expected_sensors: int) -> np.ndarray:
    """Normalize and align a sensor matrix to `(n_sensors, n_time_samples)`."""
    aligned = normalize_sensor_matrix(np.asarray(pressure, dtype=np.float64), expected_sensors=expected_sensors)
    if aligned.ndim != 2:
        raise AssertionError(f"expected a 2-D sensor matrix, got {aligned.shape}")
    if aligned.shape[1] > 1:
        aligned = aligned[:, 1:]
    return aligned


def _load_cached_result(path: os.PathLike[str] | str) -> dict[str, np.ndarray | float] | None:
    """Load a cached simulation result if the cache version matches."""
    if REFRESH_CACHE:
        return None
    cache_path = os.fspath(path)
    if not os.path.exists(cache_path):
        return None
    cached = np.load(cache_path, allow_pickle=False)
    version = int(np.asarray(cached["cache_version"]).reshape(())) if "cache_version" in cached.files else 0
    if version != CACHE_VERSION:
        return None
    return {
        "pressure": np.asarray(cached["pressure"], dtype=np.float64),
        "time": np.asarray(cached["time"], dtype=np.float64),
        "dt": float(cached["dt"]),
        "runtime_s": float(cached["runtime_s"]),
    }


def run_kwave_reference() -> dict[str, np.ndarray | float]:
    """Run the k-wave-python reference simulation."""
    cached = _load_cached_result(KWAVE_CACHE)
    if cached is not None:
        return cached

    kgrid = kWaveGrid(GRID_POINTS, GRID_SPACING)
    kgrid.makeTime(SOUND_SPEED)
    medium = kWaveMedium(sound_speed=SOUND_SPEED, density=DENSITY)

    source_p0 = _build_source_p0()
    source = kSource()
    source.p0 = source_p0

    arc_positions_kw, _ = _build_array_geometries()
    sensor_array = _build_kwave_array(arc_positions_kw)
    sensor_mask = np.asarray(sensor_array.get_array_binary_mask(kgrid), dtype=bool)
    sensor = kSensor(sensor_mask.astype(np.int32))
    sensor.record = ["p"]

    sim_options = SimulationOptions(
        pml_inside=True,
        pml_size=PML_SIZE,
        data_cast="single",
        save_to_disk=True,
        input_filename="at_array_as_sensor_kwave.h5",
        data_path=tempfile.gettempdir(),
    )
    exec_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        verbose_level=0,
        show_sim_log=False,
    )

    start = time.perf_counter()
    sensor_data = kspaceFirstOrder2D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=sim_options,
        execution_options=exec_options,
    )
    runtime_s = time.perf_counter() - start

    pressure = normalize_sensor_matrix(sensor_data["p"], expected_sensors=int(np.count_nonzero(sensor_mask)))
    result = {
        "pressure": pressure,
        "time": np.asarray(kgrid.t_array, dtype=np.float64).ravel(),
        "dt": float(kgrid.dt),
        "runtime_s": runtime_s,
    }
    np.savez(KWAVE_CACHE, cache_version=CACHE_VERSION, **result)
    return result


def run_pykwavers_reference() -> dict[str, np.ndarray | float]:
    """Run the pykwavers counterpart simulation."""
    cached = _load_cached_result(PYKWAVERS_CACHE)
    if cached is not None:
        return cached

    kgrid = kWaveGrid(GRID_POINTS, GRID_SPACING)
    kgrid.makeTime(SOUND_SPEED)

    source_p0 = _build_source_p0()
    arc_positions_kw, arc_geometries = _build_array_geometries()
    grid = pkw.Grid(nx=NX, ny=NY, nz=1, dx=DX, dy=DY, dz=DX)
    medium = pkw.Medium.homogeneous(sound_speed=SOUND_SPEED, density=DENSITY)
    source = pkw.Source.from_initial_pressure(source_p0)
    sensor_array = _build_pykwavers_array(arc_geometries)
    sensor_mask = np.squeeze(np.asarray(sensor_array.get_array_binary_mask(grid), dtype=bool))
    sensor = pkw.Sensor.from_mask(sensor_mask[:, :, None])

    start = time.perf_counter()
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)
    result = sim.run(time_steps=int(kgrid.Nt), dt=float(kgrid.dt))
    runtime_s = time.perf_counter() - start

    pressure = normalize_sensor_matrix(result.sensor_data, expected_sensors=int(np.count_nonzero(sensor_mask)))
    output = {
        "pressure": pressure,
        "time": np.asarray(result.time, dtype=np.float64).ravel(),
        "dt": float(result.dt),
        "runtime_s": runtime_s,
    }
    np.savez(
        PYKWAVERS_CACHE,
        cache_version=CACHE_VERSION,
        pressure=output["pressure"],
        time=output["time"],
        dt=output["dt"],
        runtime_s=output["runtime_s"],
    )
    return output


def run_comparison() -> dict[str, object]:
    """Run the full comparison and return metrics plus aligned matrices."""
    arc_positions_kw, arc_geometries = _build_array_geometries()
    kwave_array = _build_kwave_array(arc_positions_kw)
    pykwavers_array = _build_pykwavers_array(arc_geometries)
    grid_pk = pkw.Grid(nx=NX, ny=NY, nz=1, dx=DX, dy=DY, dz=DX)

    kw_results = run_kwave_reference()
    py_results = run_pykwavers_reference()

    kgrid = kWaveGrid(GRID_POINTS, GRID_SPACING)
    kgrid.makeTime(SOUND_SPEED)

    kw_pressure = _align_sensor_matrix(kw_results["pressure"], expected_sensors=int(np.count_nonzero(kwave_array.get_array_binary_mask(kgrid))))
    sensor_mask_pk = np.squeeze(np.asarray(pykwavers_array.get_array_binary_mask(grid_pk), dtype=bool))
    sensor_weighted_mask_pk = np.squeeze(np.asarray(pykwavers_array.get_array_weighted_mask(grid_pk), dtype=np.float64))
    py_pressure = _align_sensor_matrix(py_results["pressure"], expected_sensors=int(np.count_nonzero(sensor_mask_pk)))
    if kw_pressure.shape != py_pressure.shape:
        raise AssertionError(f"raw sensor matrix mismatch: {kw_pressure.shape} != {py_pressure.shape}")

    kw_time = np.asarray(kw_results["time"], dtype=np.float64)
    py_time = np.asarray(py_results["time"], dtype=np.float64)
    if kw_time.size > 1 and py_time.size > 1:
        kw_time = kw_time[1:]
        py_time = py_time[:-1]
    n_time = min(kw_time.size, py_time.size, kw_pressure.shape[1], py_pressure.shape[1])
    kw_time = kw_time[:n_time]
    py_time = py_time[:n_time]
    kw_pressure = kw_pressure[:, :n_time]
    py_pressure = py_pressure[:, :n_time]

    sensor_mask_kw = np.asarray(kwave_array.get_array_binary_mask(kgrid), dtype=bool)
    sensor_weighted_mask_kw = np.asarray(kwave_array.get_array_grid_weights(kgrid), dtype=np.float64)

    if sensor_mask_kw.shape != sensor_mask_pk.shape:
        raise AssertionError(f"sensor mask mismatch: {sensor_mask_kw.shape} != {sensor_mask_pk.shape}")
    if sensor_weighted_mask_kw.shape != sensor_weighted_mask_pk.shape:
        raise AssertionError(
            f"weighted sensor mask mismatch: {sensor_weighted_mask_kw.shape} != {sensor_weighted_mask_pk.shape}"
        )
    if not np.array_equal(sensor_mask_kw, sensor_mask_pk):
        raise AssertionError("binary sensor mask mismatch between k-wave-python and pykwavers")
    if not np.allclose(sensor_weighted_mask_kw, sensor_weighted_mask_pk, rtol=1e-9, atol=1e-12):
        raise AssertionError("weighted sensor mask mismatch between k-wave-python and pykwavers")

    element_weight_masks_pk = _build_pykwavers_single_element_masks(arc_geometries, grid_pk)
    element_measures_m = [geom.measure_m for geom in arc_geometries]

    kw_combined = kwave_array.combine_sensor_data(kgrid, kw_pressure, order="F")
    py_combined = combine_array_sensor_data(
        py_pressure,
        sensor_mask_pk,
        element_weight_masks_pk,
        element_measures_m,
        grid_spacing_m=DX,
        element_dimension=1,
    )

    if kw_combined.shape != py_combined.shape:
        raise AssertionError(f"combined trace mismatch: {kw_combined.shape} != {py_combined.shape}")

    mask_metrics = compute_image_metrics(sensor_mask_kw.astype(np.float64), sensor_mask_pk.astype(np.float64))
    weighted_mask_metrics = compute_image_metrics(sensor_weighted_mask_kw, sensor_weighted_mask_pk)
    raw_matrix_metrics = compute_image_metrics(kw_pressure, py_pressure)
    combined_matrix_metrics = compute_image_metrics(kw_combined, py_combined)

    trace_metrics = {
        f"element_{idx:02d}": compute_trace_metrics(kw_combined[idx], py_combined[idx])
        for idx in range(kw_combined.shape[0])
    }
    trace_summary = {
        "pearson_r_mean": float(np.mean([m["pearson_r"] for m in trace_metrics.values()])),
        "pearson_r_min": float(np.min([m["pearson_r"] for m in trace_metrics.values()])),
        "pearson_r_median": float(np.median([m["pearson_r"] for m in trace_metrics.values()])),
        "rms_ratio_mean": float(np.mean([m["rms_ratio"] for m in trace_metrics.values()])),
        "rms_ratio_min": float(np.min([m["rms_ratio"] for m in trace_metrics.values()])),
        "rms_ratio_max": float(np.max([m["rms_ratio"] for m in trace_metrics.values()])),
        "rms_ratio_median": float(np.median([m["rms_ratio"] for m in trace_metrics.values()])),
        "rmse_mean": float(np.mean([m["rmse"] for m in trace_metrics.values()])),
        "rmse_max": float(np.max([m["rmse"] for m in trace_metrics.values()])),
        "rmse_median": float(np.median([m["rmse"] for m in trace_metrics.values()])),
        "peak_ratio_mean": float(np.mean([m["peak_ratio"] for m in trace_metrics.values()])),
        "peak_ratio_min": float(np.min([m["peak_ratio"] for m in trace_metrics.values()])),
        "peak_ratio_max": float(np.max([m["peak_ratio"] for m in trace_metrics.values()])),
        "peak_ratio_median": float(np.median([m["peak_ratio"] for m in trace_metrics.values()])),
    }

    return {
        "kwave": {
            "pressure": kw_pressure,
            "combined": kw_combined,
            "time": kw_time,
            "dt": float(kw_results["dt"]),
            "runtime_s": float(kw_results["runtime_s"]),
        },
        "pykwavers": {
            "pressure": py_pressure,
            "combined": py_combined,
            "time": py_time,
            "dt": float(py_results["dt"]),
            "runtime_s": float(py_results["runtime_s"]),
        },
        "layout": {
            "source_p0": _build_source_p0(),
            "sensor_mask_kw": sensor_mask_kw,
            "sensor_mask_pk": sensor_mask_pk,
            "sensor_weighted_mask_kw": sensor_weighted_mask_kw,
            "sensor_weighted_mask_pk": sensor_weighted_mask_pk,
        },
        "mask_metrics": mask_metrics,
        "weighted_mask_metrics": weighted_mask_metrics,
        "raw_matrix_metrics": raw_matrix_metrics,
        "combined_matrix_metrics": combined_matrix_metrics,
        "trace_metrics": trace_metrics,
        "trace_summary": trace_summary,
    }


def plot_comparison(result: dict[str, object]) -> None:
    """Render the source layout, raw matrices, and combined traces."""
    kwave = result["kwave"]  # type: ignore[assignment]
    pykwavers = result["pykwavers"]  # type: ignore[assignment]
    layout = result["layout"]  # type: ignore[assignment]

    source_p0 = np.asarray(layout["source_p0"], dtype=np.float64)  # type: ignore[index]
    sensor_mask_kw = np.asarray(layout["sensor_mask_kw"], dtype=bool)  # type: ignore[index]
    sensor_mask_pk = np.asarray(layout["sensor_mask_pk"], dtype=bool)  # type: ignore[index]
    kw_pressure = np.asarray(kwave["pressure"], dtype=np.float64)  # type: ignore[index]
    py_pressure = np.asarray(pykwavers["pressure"], dtype=np.float64)  # type: ignore[index]
    kw_combined = np.asarray(kwave["combined"], dtype=np.float64)  # type: ignore[index]
    py_combined = np.asarray(pykwavers["combined"], dtype=np.float64)  # type: ignore[index]
    time_us = np.asarray(kwave["time"], dtype=np.float64) * 1.0e6  # type: ignore[index]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    ax.imshow(source_p0.T, origin="lower", cmap="magma", interpolation="nearest")
    ax.contour(sensor_mask_kw.T.astype(np.float64), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.contour(sensor_mask_pk.T.astype(np.float64), levels=[0.5], colors="white", linewidths=0.5, linestyles="--")
    ax.set_title("Initial pressure source and detector ring")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")

    ax = axes[0, 1]
    im = ax.imshow(
        kw_pressure - py_pressure,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_title("Raw detector matrix difference")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Detector point")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.plot(time_us, kw_combined[0], color="black", linewidth=1.5, label="k-wave-python")
    ax.plot(time_us, py_combined[0], color="tab:red", linewidth=1.1, linestyle="--", label="pykwavers")
    ax.set_title("Combined trace for detector 0")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Pressure [Pa]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    im = ax.imshow(
        kw_combined - py_combined,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_title("Combined arc-trace difference")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Detector element")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("at_array_as_sensor: k-wave-python vs pykwavers", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_report_lines(result: dict[str, object]) -> list[str]:
    """Construct the plain-text metrics report."""
    kwave = result["kwave"]  # type: ignore[assignment]
    pykwavers = result["pykwavers"]  # type: ignore[assignment]

    return [
        "example: at_array_as_sensor",
        f"grid_points: {NX}x{NY}",
        f"dt_s: {float(kwave['dt']):.9e}",
        f"kwave_runtime_s: {float(kwave['runtime_s']):.3f}",
        f"pykwavers_runtime_s: {float(pykwavers['runtime_s']):.3f}",
        "",
        "sensor mask parity:",
        f"  pearson_r = {result['mask_metrics']['pearson_r']:.6f}",
        f"  rms_ratio = {result['mask_metrics']['rms_ratio']:.6f}",
        f"  rmse      = {result['mask_metrics']['rmse']:.6e}",
        f"  max_abs_diff = {result['mask_metrics']['max_abs_diff']:.6e}",
        f"  peak_ratio= {result['mask_metrics']['peak_ratio']:.6f}",
        f"  psnr_db   = {result['mask_metrics']['psnr_db']:.6f}",
        "",
        "weighted mask parity:",
        f"  pearson_r = {result['weighted_mask_metrics']['pearson_r']:.6f}",
        f"  rms_ratio = {result['weighted_mask_metrics']['rms_ratio']:.6f}",
        f"  rmse      = {result['weighted_mask_metrics']['rmse']:.6e}",
        f"  max_abs_diff = {result['weighted_mask_metrics']['max_abs_diff']:.6e}",
        f"  peak_ratio= {result['weighted_mask_metrics']['peak_ratio']:.6f}",
        f"  psnr_db   = {result['weighted_mask_metrics']['psnr_db']:.6f}",
        "",
        "raw detector matrix parity:",
        f"  pearson_r = {result['raw_matrix_metrics']['pearson_r']:.6f}",
        f"  rms_ratio = {result['raw_matrix_metrics']['rms_ratio']:.6f}",
        f"  rmse      = {result['raw_matrix_metrics']['rmse']:.6e}",
        f"  max_abs_diff = {result['raw_matrix_metrics']['max_abs_diff']:.6e}",
        f"  peak_ratio= {result['raw_matrix_metrics']['peak_ratio']:.6f}",
        f"  psnr_db   = {result['raw_matrix_metrics']['psnr_db']:.6f}",
        "",
        "combined arc-trace parity:",
        f"  pearson_r = {result['combined_matrix_metrics']['pearson_r']:.6f}",
        f"  rms_ratio = {result['combined_matrix_metrics']['rms_ratio']:.6f}",
        f"  rmse      = {result['combined_matrix_metrics']['rmse']:.6e}",
        f"  max_abs_diff = {result['combined_matrix_metrics']['max_abs_diff']:.6e}",
        f"  peak_ratio= {result['combined_matrix_metrics']['peak_ratio']:.6f}",
        f"  psnr_db   = {result['combined_matrix_metrics']['psnr_db']:.6f}",
        "",
        f"combined trace pearson_r_min = {result['trace_summary']['pearson_r_min']:.6f}",
        f"combined trace pearson_r_mean = {result['trace_summary']['pearson_r_mean']:.6f}",
        f"combined trace rms_ratio_min = {result['trace_summary']['rms_ratio_min']:.6f}",
        f"combined trace rms_ratio_max = {result['trace_summary']['rms_ratio_max']:.6f}",
        f"combined trace rms_ratio_mean = {result['trace_summary']['rms_ratio_mean']:.6f}",
        f"combined trace peak_ratio_min = {result['trace_summary']['peak_ratio_min']:.6f}",
        f"combined trace peak_ratio_max = {result['trace_summary']['peak_ratio_max']:.6f}",
        f"combined trace rmse_max = {result['trace_summary']['rmse_max']:.6e}",
    ]


PARITY_THRESHOLDS = {
    "mask_metrics": {
        "pearson_r": 0.99999,
    },
    "weighted_mask_metrics": {
        "pearson_r": 0.99999,
    },
    "raw_matrix_metrics": {
        "pearson_r": 0.98,
        "psnr_db": 30.0,
    },
    "combined_matrix_metrics": {
        "pearson_r": 0.99,
        "psnr_db": 30.0,
    },
    "trace_summary": {
        "pearson_r_min": 0.99,
        "pearson_r_mean": 0.99,
        "rms_ratio_min": 1.0 - 3e-2,
        "rms_ratio_max": 1.0 + 3e-2,
        "peak_ratio_min": 1.0 - 1.5e-1,
        "peak_ratio_max": 1.0 + 1.5e-1,
        "rmse_max": 2e-2,
    },
}


def _metrics_pass(result: dict[str, object]) -> bool:
    """Return true when metrics satisfy the script-owned parity contract."""
    for metric_name in (
        "mask_metrics",
        "weighted_mask_metrics",
        "raw_matrix_metrics",
        "combined_matrix_metrics",
    ):
        metrics = result[metric_name]  # type: ignore[index]
        thresholds = PARITY_THRESHOLDS[metric_name]
        if metrics["pearson_r"] < thresholds["pearson_r"]:  # type: ignore[index]
            return False
        if "psnr_db" in thresholds and metrics["psnr_db"] <= thresholds["psnr_db"]:  # type: ignore[index]
            return False

    summary = result["trace_summary"]  # type: ignore[index]
    thresholds = PARITY_THRESHOLDS["trace_summary"]
    return (
        summary["pearson_r_min"] >= thresholds["pearson_r_min"]  # type: ignore[index]
        and summary["pearson_r_mean"] >= thresholds["pearson_r_mean"]  # type: ignore[index]
        and summary["rms_ratio_min"] >= thresholds["rms_ratio_min"]  # type: ignore[index]
        and summary["rms_ratio_max"] <= thresholds["rms_ratio_max"]  # type: ignore[index]
        and summary["peak_ratio_min"] >= thresholds["peak_ratio_min"]  # type: ignore[index]
        and summary["peak_ratio_max"] <= thresholds["peak_ratio_max"]  # type: ignore[index]
        and summary["rmse_max"] < thresholds["rmse_max"]  # type: ignore[index]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="k-wave-python at_array_as_sensor parity example")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute both reference simulations and overwrite the caches.",
    )
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    global REFRESH_CACHE
    if args.no_cache:
        REFRESH_CACHE = True

    result = run_comparison()

    r_min = float(result["trace_summary"]["pearson_r_min"])
    rms_mean = float(result["trace_summary"]["rms_ratio_mean"])
    overall_status = "PASS" if _metrics_pass(result) else "FAIL"

    report_lines = build_report_lines(result)
    report_lines.append(f"parity_status: {overall_status}")
    save_text_report(METRICS_PATH, "at_array_as_sensor parity metrics", report_lines)

    plot_comparison(result)

    print("=" * 80)
    print("k-wave-python at_array_as_sensor vs pykwavers")
    print("=" * 80)
    print(f"Sensor mask pearson r:       {result['mask_metrics']['pearson_r']:.6f}")
    print(f"Weighted mask pearson r:     {result['weighted_mask_metrics']['pearson_r']:.6f}")
    print(f"Raw detector matrix pearson: {result['raw_matrix_metrics']['pearson_r']:.6f}")
    print(f"Combined trace pearson:      {result['combined_matrix_metrics']['pearson_r']:.6f}")
    print(
        "Combined trace min corr:     "
        f"{r_min:.6f}  (target >= {PARITY_THRESHOLDS['trace_summary']['pearson_r_min']})"
    )
    print(
        "Combined trace rms_ratio:    "
        f"{rms_mean:.6f}  "
        f"(target [{PARITY_THRESHOLDS['trace_summary']['rms_ratio_min']}, "
        f"{PARITY_THRESHOLDS['trace_summary']['rms_ratio_max']}])"
    )
    print(f"k-Wave runtime [s]:          {float(result['kwave']['runtime_s']):.3f}")
    print(f"pykwavers runtime [s]:       {float(result['pykwavers']['runtime_s']):.3f}")
    print(f"Status:                      {overall_status}")

    return 0 if overall_status == "PASS" or args.allow_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
