"""
Comparison: k-wave-python pr_2D_FFT_line_sensor vs pykwavers.

This script reproduces the vendored k-wave-python 2D line-sensor example with
the same initial-pressure source, PML settings, and sensor mask, then compares
the reconstructed initial-pressure images produced by the vendored and
pykwavers pipelines.
"""

from __future__ import annotations

import os
import time

import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    expand_pml_outside_shape,
    compute_image_metrics,
    pad_volume_for_pml_outside,
    compute_trace_metrics,
    normalize_sensor_matrix,
)


_ROOT = bootstrap_example_paths()

KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "pr_2D_FFT_line_sensor_kwave_cache.npz"
PYKWAVERS_CACHE = DEFAULT_OUTPUT_DIR / "pr_2D_FFT_line_sensor_pykwavers_cache.npz"
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 4

import pykwavers as kw

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceLineRecon import kspaceLineRecon as kwave_kspace_line_recon
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc
from scipy.interpolate import RegularGridInterpolator


def _build_example_inputs() -> dict[str, object]:
    """Construct the exact example configuration shared by both solvers."""
    pml_size = Vector([20, 20])
    grid_points_vec = Vector([128, 256]) - 2 * pml_size
    grid_points = tuple(int(v) for v in grid_points_vec)
    spacing = Vector([0.1e-3, 0.1e-3])
    sound_speed = 1500.0

    medium = kWaveMedium(sound_speed=sound_speed)

    disc_2 = 5.0 * make_disc(grid_points_vec, Vector([60, 140]), 5)
    disc_1 = 5.0 * make_disc(grid_points_vec, Vector([30, 110]), 8)
    source = kSource()
    source.p0 = smooth(disc_1 + disc_2, True)

    sensor_mask = np.zeros(grid_points, dtype=bool)
    sensor_mask[0, :] = True
    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]

    return {
        "grid_points": grid_points,
        "spacing": tuple(float(v) for v in spacing),
        "sound_speed": sound_speed,
        "pml_size": tuple(int(v) for v in pml_size),
        "medium": medium,
        "source": source,
        "sensor": sensor,
        "sensor_mask": sensor_mask,
    }


def _build_kgrid(grid_points: tuple[int, int], spacing: tuple[float, float], sound_speed: float) -> kWaveGrid:
    kgrid = kWaveGrid(Vector(list(grid_points)), Vector(list(spacing)))
    kgrid.makeTime(sound_speed)
    return kgrid


def _reconstruct_line_sensor_kwave(
    pressure: np.ndarray,
    *,
    spacing: tuple[float, float],
    dt: float,
    sound_speed: float,
) -> np.ndarray:
    """Run the vendored k-Wave line reconstruction on `(sensor, time)` data."""
    pressure_ty = np.asarray(pressure, dtype=np.float64).T
    recon = kwave_kspace_line_recon(
        pressure_ty,
        dy=float(spacing[1]),
        dt=float(dt),
        c=float(sound_speed),
        data_order="ty",
        interp="linear",
        pos_cond=True,
    )
    return np.asarray(recon, dtype=np.float64)


def _reconstruct_line_sensor_pykwavers(
    pressure: np.ndarray,
    *,
    spacing: tuple[float, float],
    dt: float,
    sound_speed: float,
) -> np.ndarray:
    """Run the native pykwavers line reconstruction on `(sensor, time)` data."""
    pressure_ty = np.asarray(pressure, dtype=np.float64).T
    recon = kw.kspace_line_recon(
        pressure_ty,
        dy=float(spacing[1]),
        dt=float(dt),
        c=float(sound_speed),
        data_order="ty",
        interp="linear",
        pos_cond=True,
    )
    return np.asarray(recon, dtype=np.float64)


def _resample_reconstruction_to_source_grid(
    reconstruction: np.ndarray,
    *,
    kgrid: kWaveGrid,
    dt: float,
    sound_speed: float,
) -> np.ndarray:
    """Resample a line reconstruction onto the original source grid."""
    recon_grid = kWaveGrid(Vector(list(reconstruction.shape)), Vector([dt * sound_speed, kgrid.dy]))
    interp_func = RegularGridInterpolator(
        (recon_grid.x_vec[:, 0] - recon_grid.x_vec.min(), recon_grid.y_vec[:, 0]),
        reconstruction,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    query_points = np.stack((kgrid.x - kgrid.x.min(), kgrid.y), axis=-1)
    return np.asarray(interp_func(query_points), dtype=np.float64)


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

    inputs = _build_example_inputs()
    grid_points = inputs["grid_points"]
    spacing = inputs["spacing"]
    sound_speed = float(inputs["sound_speed"])
    pml_size = inputs["pml_size"]
    medium = inputs["medium"]
    source = inputs["source"]
    sensor = inputs["sensor"]

    kgrid = _build_kgrid(grid_points, spacing, sound_speed)

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
    sensor_data = kspaceFirstOrder2D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=sim_options,
        execution_options=exec_options,
    )
    runtime = time.perf_counter() - start

    pressure = np.asarray(sensor_data["p"], dtype=np.float64)
    pressure = normalize_sensor_matrix(pressure, expected_sensors=int(grid_points[1]))
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

    inputs = _build_example_inputs()
    grid_points = inputs["grid_points"]
    spacing = inputs["spacing"]
    sound_speed = float(inputs["sound_speed"])
    medium = inputs["medium"]
    source = inputs["source"]
    sensor_mask = inputs["sensor_mask"]
    pml_size = tuple(int(v) for v in inputs["pml_size"])
    expanded_grid_points = expand_pml_outside_shape(grid_points, pml_size)
    expanded_source_p0 = pad_volume_for_pml_outside(np.asarray(source.p0, dtype=np.float64), pml_size)
    expanded_sensor_mask = pad_volume_for_pml_outside(sensor_mask.astype(bool), pml_size).astype(bool)

    kgrid = _build_kgrid(grid_points, spacing, sound_speed)

    grid = kw.Grid(
        nx=expanded_grid_points[0],
        ny=expanded_grid_points[1],
        nz=expanded_grid_points[2],
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[0],
    )
    medium_pk = kw.Medium.homogeneous(sound_speed=float(medium.sound_speed), density=1000.0)
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
    pressure = normalize_sensor_matrix(pressure, expected_sensors=int(grid_points[1]))
    result = {
        "pressure": pressure,
        "time": np.asarray(result.time, dtype=np.float64),
        "dt": float(result.dt),
        "runtime": runtime,
    }
    np.savez(PYKWAVERS_CACHE, cache_version=CACHE_VERSION, **result)
    return result


def run_comparison() -> dict[str, object]:
    """Run the comparison and return reconstruction metrics plus raw matrices."""
    inputs = _build_example_inputs()
    source_pressure = np.asarray(inputs["source"].p0, dtype=np.float64)
    grid_points = inputs["grid_points"]
    spacing = inputs["spacing"]
    sound_speed = float(inputs["sound_speed"])
    kw_results = run_kwave_reference()
    py_results = run_pykwavers_reference()

    kw_pressure = np.asarray(kw_results["pressure"], dtype=np.float64)
    py_pressure = np.asarray(py_results["pressure"], dtype=np.float64)
    if kw_pressure.shape != py_pressure.shape:
        raise AssertionError(f"Sensor matrix shape mismatch: {kw_pressure.shape} != {py_pressure.shape}")

    if kw_pressure.shape[1] > 1 and py_pressure.shape[1] > 1:
        kw_pressure = kw_pressure[:, 1:]
        py_pressure = py_pressure[:, :-1]
        kw_results = {
            **kw_results,
            "pressure": kw_pressure,
            "time": np.asarray(kw_results["time"], dtype=np.float64)[1:],
        }
        py_results = {
            **py_results,
            "pressure": py_pressure,
            "time": np.asarray(py_results["time"], dtype=np.float64)[:-1],
        }

    kgrid = _build_kgrid(grid_points, spacing, sound_speed)
    reference_pressure = kw_pressure
    kw_reconstruction = _resample_reconstruction_to_source_grid(
        _reconstruct_line_sensor_kwave(
            reference_pressure,
            spacing=spacing,
            dt=float(kw_results["dt"]),
            sound_speed=sound_speed,
        ),
        kgrid=kgrid,
        dt=float(kw_results["dt"]),
        sound_speed=sound_speed,
    )
    py_reconstruction = _resample_reconstruction_to_source_grid(
        _reconstruct_line_sensor_pykwavers(
            reference_pressure,
            spacing=spacing,
            dt=float(py_results["dt"]),
            sound_speed=sound_speed,
        ),
        kgrid=kgrid,
        dt=float(py_results["dt"]),
        sound_speed=sound_speed,
    )

    kw_results = {**kw_results, "pressure": kw_pressure, "reconstruction": kw_reconstruction}
    py_results = {**py_results, "pressure": py_pressure, "reconstruction": py_reconstruction}
    if kw_reconstruction.shape != py_reconstruction.shape:
        raise AssertionError(
            f"Reconstruction shape mismatch: {kw_reconstruction.shape} != {py_reconstruction.shape}"
        )

    summary = compute_image_metrics(kw_reconstruction, py_reconstruction)
    reference_metrics = {
        "kwave": compute_image_metrics(source_pressure, kw_reconstruction),
        "pykwavers": compute_image_metrics(source_pressure, py_reconstruction),
    }

    representative_rows = [0, kw_pressure.shape[0] // 2, kw_pressure.shape[0] - 1]
    trace_metrics = {
        row: compute_trace_metrics(kw_pressure[row], py_pressure[row])
        for row in dict.fromkeys(representative_rows)
    }

    return {
        "kwave": kw_results,
        "pykwavers": py_results,
        "summary": summary,
        "reference_metrics": reference_metrics,
        "trace_metrics": trace_metrics,
    }


def main() -> int:
    """Execute the comparison and print metric summaries."""
    result = run_comparison()
    summary = result["summary"]

    print("=" * 80)
    print("k-wave-python pr_2D_FFT_line_sensor vs pykwavers")
    print("=" * 80)
    print(f"Image Pearson r:  {summary['pearson_r']:.6f}")
    print(f"Image RMS ratio:  {summary['rms_ratio']:.6f}")
    print(f"Image PSNR [dB]:   {summary['psnr_db']:.6f}")
    print(f"k-Wave vs p0 r:    {result['reference_metrics']['kwave']['pearson_r']:.6f}")
    print(f"pykwavers vs p0 r: {result['reference_metrics']['pykwavers']['pearson_r']:.6f}")

    for row, metrics in result["trace_metrics"].items():
        print(
            f"Sensor row {row}: corr={metrics['pearson_r']:.6f}, "
            f"rmse={metrics['rmse']:.6e}, peak_ratio={metrics['peak_ratio']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
