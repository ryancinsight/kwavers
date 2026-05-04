"""
Comparison: k-wave-python pr_3D_FFT_planar_sensor vs pykwavers.

This script reproduces the vendored k-wave-python 3D planar-sensor example
with the same initial-pressure source, PML settings, and sensor mask, then
compares the recorded pressure matrix against pykwavers using trace metrics.
"""

from __future__ import annotations

import os
import time

import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    expand_pml_outside_shape,
    pad_volume_for_pml_outside,
    compute_trace_metrics,
    normalize_sensor_matrix,
    save_side_by_side_parity_figure,
    summarize_sensor_matrix_metrics,
)


_ROOT = bootstrap_example_paths()

KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "pr_3D_FFT_planar_sensor_kwave_cache.npz"
PYKWAVERS_CACHE = DEFAULT_OUTPUT_DIR / "pr_3D_FFT_planar_sensor_pykwavers_cache.npz"
PRESSURE_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "pr_3D_FFT_planar_sensor_pressure_compare.png"
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 2

import pykwavers as kw

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball


def _build_example_inputs(scale: int = 1) -> dict[str, object]:
    """Construct the exact example configuration shared by both solvers."""
    pml_size = Vector([10, 10, 10])
    grid_points_vec = scale * Vector([32, 64, 64]) - 2 * pml_size
    grid_points = tuple(int(v) for v in grid_points_vec)
    spacing = 1e-3 * Vector([0.2, 0.2, 0.2]) / scale

    medium = kWaveMedium(sound_speed=1500)

    p0 = 10.0 * make_ball(grid_points_vec, grid_points_vec / 2, 3 * scale)
    source = kSource()
    source.p0 = smooth(p0, True)

    sensor_mask = np.zeros(grid_points, dtype=bool)
    sensor_mask[0, :, :] = True
    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]

    return {
        "grid_points": grid_points,
        "spacing": tuple(float(v) for v in spacing),
        "pml_size": tuple(int(v) for v in pml_size),
        "medium": medium,
        "source": source,
        "sensor": sensor,
        "sensor_mask": sensor_mask,
    }


def _load_cached_result(path: os.PathLike[str] | str) -> dict[str, np.ndarray | float] | None:
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
        "runtime": float(cached["runtime"]),
    }


def run_kwave_reference() -> dict[str, np.ndarray | float]:
    """Run the k-wave-python reference simulation."""
    cached = _load_cached_result(KWAVE_CACHE)
    if cached is not None:
        return cached

    inputs = _build_example_inputs(scale=1)
    grid_points = inputs["grid_points"]
    spacing = inputs["spacing"]
    pml_size = inputs["pml_size"]
    medium = inputs["medium"]
    source = inputs["source"]
    sensor = inputs["sensor"]

    kgrid = kWaveGrid(Vector(list(grid_points)), Vector(list(spacing)))
    kgrid.makeTime(medium.sound_speed)

    sim_options = SimulationOptions(
        pml_inside=False,
        pml_size=Vector(list(pml_size)),
        smooth_p0=False,
        save_to_disk=True,
    )
    exec_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        verbose_level=0,
        show_sim_log=False,
    )

    start = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=sim_options,
        execution_options=exec_options,
    )
    runtime = time.perf_counter() - start

    pressure = np.asarray(sensor_data["p"], dtype=np.float64)
    pressure = normalize_sensor_matrix(pressure, expected_sensors=int(np.prod(grid_points[1:])))
    result = {
        "pressure": pressure,
        "time": np.asarray(kgrid.t_array, dtype=np.float64),
        "dt": float(kgrid.dt),
        "runtime": runtime,
    }
    np.savez(KWAVE_CACHE, cache_version=CACHE_VERSION, **result)
    return result


def run_pykwavers_reference() -> dict[str, np.ndarray | float]:
    """Run the equivalent pykwavers simulation."""
    cached = _load_cached_result(PYKWAVERS_CACHE)
    if cached is not None:
        return cached

    inputs = _build_example_inputs(scale=1)
    grid_points = inputs["grid_points"]
    spacing = inputs["spacing"]
    medium = inputs["medium"]
    source = inputs["source"]
    sensor_mask = inputs["sensor_mask"]
    pml_size = tuple(int(v) for v in inputs["pml_size"])
    expanded_grid_points = expand_pml_outside_shape(grid_points, pml_size)
    expanded_source_p0 = pad_volume_for_pml_outside(np.asarray(source.p0, dtype=np.float64), pml_size)
    expanded_sensor_mask = pad_volume_for_pml_outside(sensor_mask.astype(bool), pml_size).astype(bool)

    kgrid = kWaveGrid(Vector(list(grid_points)), Vector(list(spacing)))
    kgrid.makeTime(medium.sound_speed)

    grid = kw.Grid(
        nx=expanded_grid_points[0],
        ny=expanded_grid_points[1],
        nz=expanded_grid_points[2],
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[2],
    )
    medium_pk = kw.Medium.homogeneous(sound_speed=float(np.asarray(medium.sound_speed).flat[0]), density=1000.0)
    source_pk = kw.Source.from_initial_pressure(expanded_source_p0)
    sensor_pk = kw.Sensor.from_mask(expanded_sensor_mask)

    sim = kw.Simulation(
        grid,
        medium_pk,
        source_pk,
        sensor_pk,
        solver=kw.SolverType.PSTD,
        pml_size=pml_size[0],
    )
    sim.set_pml_inside(True)

    start = time.perf_counter()
    result = sim.run(time_steps=int(kgrid.Nt), dt=float(kgrid.dt))
    runtime = time.perf_counter() - start

    pressure = np.asarray(result.sensor_data, dtype=np.float64)
    pressure = normalize_sensor_matrix(pressure, expected_sensors=int(np.prod(grid_points[1:])))
    result = {
        "pressure": pressure,
        "time": np.asarray(result.time, dtype=np.float64),
        "dt": float(result.dt),
        "runtime": runtime,
    }
    np.savez(PYKWAVERS_CACHE, cache_version=CACHE_VERSION, **result)
    return result


def run_comparison() -> dict[str, object]:
    """Run the comparison and return metrics plus raw matrices."""
    kw_results = run_kwave_reference()
    py_results = run_pykwavers_reference()

    kw_pressure = np.asarray(kw_results["pressure"], dtype=np.float64)
    py_pressure = np.asarray(py_results["pressure"], dtype=np.float64)
    if kw_pressure.shape != py_pressure.shape:
        raise AssertionError(f"Sensor matrix shape mismatch: {kw_pressure.shape} != {py_pressure.shape}")

    if kw_pressure.shape[1] > 1 and py_pressure.shape[1] > 1:
        kw_pressure = kw_pressure[:, 1:]
        py_pressure = py_pressure[:, :-1]
        kw_results = {**kw_results, "pressure": kw_pressure, "time": np.asarray(kw_results["time"], dtype=np.float64)[1:]}
        py_results = {**py_results, "pressure": py_pressure, "time": np.asarray(py_results["time"], dtype=np.float64)[:-1]}

    summary = summarize_sensor_matrix_metrics(
        kw_pressure,
        py_pressure,
        expected_sensors=kw_pressure.shape[0],
    )

    representative_rows = [0, kw_pressure.shape[0] // 2, kw_pressure.shape[0] - 1]
    trace_metrics = {
        row: compute_trace_metrics(kw_pressure[row], py_pressure[row])
        for row in dict.fromkeys(representative_rows)
    }

    return {
        "kwave": kw_results,
        "pykwavers": py_results,
        "summary": summary,
        "trace_metrics": trace_metrics,
    }


_R_TARGET = 0.98
_RMS_MIN = 0.90
_RMS_MAX = 1.10


def main() -> int:
    """Execute the comparison, print metric summaries, and write a metrics file."""
    result = run_comparison()
    summary = result["summary"]

    r_mean = float(summary["pearson_r_mean"])
    rms_mean = float(summary["rms_ratio_mean"])
    checks = {
        "pearson_r": r_mean >= _R_TARGET,
        "rms_ratio": _RMS_MIN <= rms_mean <= _RMS_MAX,
    }
    overall_status = "PASS" if all(checks.values()) else "FAIL"

    print("=" * 80)
    print("k-wave-python pr_3D_FFT_planar_sensor vs pykwavers")
    print("=" * 80)
    print(f"Mean Pearson r:   {r_mean:.6f}  (target >= {_R_TARGET})")
    print(f"Mean RMS ratio:   {rms_mean:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])")
    print(f"Median Pearson r: {summary['pearson_r_median']:.6f}")
    print(f"Median RMS ratio: {summary['rms_ratio_median']:.6f}")
    print(f"Median RMSE:      {summary['rmse_median']:.6e}")
    print(f"Max |diff|:       {summary['max_abs_diff_max']:.6e}")
    print(f"Median peak ratio:{summary['peak_ratio_median']:.6f}")
    print(f"Status:           {overall_status}")

    for row, metrics in result["trace_metrics"].items():
        print(f"Sensor row {row}: corr={metrics['pearson_r']:.6f}, rmse={metrics['rmse']:.6e}, peak_ratio={metrics['peak_ratio']:.6f}")

    # --- Structured metrics file ---
    kw_rt = float(result["kwave"]["runtime"])
    py_rt = float(result["pykwavers"]["runtime"])
    n_sensors = int(summary["n_sensors"])
    output_path = DEFAULT_OUTPUT_DIR / "pr_3D_FFT_planar_sensor_metrics.txt"
    pressure_figure_path = save_side_by_side_parity_figure(
        result["kwave"]["pressure"],
        result["pykwavers"]["pressure"],
        PRESSURE_FIGURE_PATH,
        title="pr_3D_FFT_planar_sensor forward sensor parity",
        reference_label="k-wave-python pressure",
        candidate_label="pykwavers pressure",
        cmap="seismic",
    )
    with output_path.open("w") as fh:
        fh.write("pr_3D_FFT_planar_sensor parity metrics\n")
        fh.write(f"parity_status: {overall_status}\n\n")
        fh.write(f"kwave_runtime_s: {kw_rt:.3f}\n")
        fh.write(f"pykwavers_runtime_s: {py_rt:.3f}\n\n")
        fh.write(f"Forward sensor matrix ({n_sensors} sensors):\n")
        fh.write(f"  pearson_r_mean    = {r_mean:.6f}  (target >= {_R_TARGET})\n")
        fh.write(f"  rms_ratio_mean    = {rms_mean:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])\n")
        fh.write(f"  pearson_r_median  = {summary['pearson_r_median']:.6f}\n")
        fh.write(f"  rms_ratio_median  = {summary['rms_ratio_median']:.6f}\n")
        fh.write(f"  rmse_median       = {summary['rmse_median']:.6e}\n")
        fh.write(f"  max_abs_diff      = {summary['max_abs_diff_max']:.6e}\n\n")
        fh.write("Sensor row traces:\n")
        for row, m in result["trace_metrics"].items():
            fh.write(
                f"  row={row}: pearson_r={m['pearson_r']:.6f}  "
                f"rms_ratio={m['rms_ratio']:.6f}  rmse={m['rmse']:.6e}\n"
            )
        fh.write(f"\nfigure_pressure: {pressure_figure_path.name}\n")
    print(f"Metrics written to: {output_path}")
    print(f"Figure written to: {pressure_figure_path}")

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
