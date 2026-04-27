"""
Comparison: k-wave-python pr_3D_TR_planar_sensor vs pykwavers.

This script reproduces the vendored 3D planar-sensor photoacoustic example,
then compares:
  1. the forward sensor matrices produced by k-wave-python and pykwavers
  2. the k-wave-python time-reversal reconstruction against the pykwavers
     native time-reversal binding on the same recorded sensor data

The reconstruction comparisons reuse the forward k-wave sensor matrix so the
metrics isolate reconstruction parity rather than forward-solver drift.
"""

from __future__ import annotations

from copy import deepcopy
import os
import time

import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    compute_trace_metrics,
    expand_pml_outside_shape,
    normalize_sensor_matrix,
    pad_volume_for_pml_outside,
)


_ROOT = bootstrap_example_paths()

KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "pr_3D_TR_planar_sensor_kwave_cache.npz"
PYKWAVERS_CACHE = DEFAULT_OUTPUT_DIR / "pr_3D_TR_planar_sensor_pykwavers_cache.npz"
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1

import pykwavers as kw

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.reconstruction import TimeReversal
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball


def _build_example_inputs(scale: int = 1) -> dict[str, object]:
    """Construct the exact example configuration shared by both solvers."""
    pml_size = Vector([10, 10, 10])
    grid_points_vec = scale * Vector([24, 40, 40]) - 2 * pml_size
    grid_points = tuple(int(v) for v in grid_points_vec)
    spacing = 1e-3 * Vector([0.2, 0.2, 0.2]) / scale

    medium = kWaveMedium(sound_speed=1500)

    p0 = 10.0 * make_ball(grid_points_vec, grid_points_vec / 2, 2 * scale)
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


def _sound_speed_scalar(medium: kWaveMedium) -> float:
    """Return the medium sound speed as a plain Python scalar."""
    return float(np.asarray(medium.sound_speed, dtype=np.float64).reshape(()))


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
        **(
            {"time_reversal": np.asarray(cached["time_reversal"], dtype=np.float64)}
            if "time_reversal" in cached.files
            else {}
        ),
    }


def _align_sensor_matrix(pressure: np.ndarray) -> np.ndarray:
    """Match the sensor-time alignment used by the FFT parity example."""
    aligned = np.asarray(pressure, dtype=np.float64)
    if aligned.ndim != 2:
        raise AssertionError(f"expected a 2-D sensor matrix, got {aligned.shape}")
    if aligned.shape[1] > 1:
        aligned = aligned[:, 1:]
    return aligned


def _mask_positions_fortran(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Return Cartesian positions for active voxels in Fortran sensor order."""
    mask_bool = np.asarray(mask, dtype=bool)
    active = np.flatnonzero(mask_bool.flatten(order="F"))
    coords = np.column_stack(np.unravel_index(active, mask_bool.shape, order="F"))
    return coords.astype(np.float64) * np.asarray(spacing, dtype=np.float64)


def _reconstruct_time_reversal_kwave(
    *,
    kgrid: kWaveGrid,
    medium: kWaveMedium,
    sensor: kSensor,
    pressure: np.ndarray,
    pml_size: tuple[int, int, int],
) -> np.ndarray:
    """Reconstruct the source image using vendored k-wave-python time reversal."""
    sensor_recon = deepcopy(sensor)
    sensor_recon.recorded_pressure = np.asarray(pressure, dtype=np.float64)

    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=Vector(list(pml_size)),
        smooth_p0=False,
        save_to_disk=True,
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        verbose_level=0,
        show_sim_log=False,
    )

    tr = TimeReversal(kgrid, medium, sensor_recon)
    p0_recon = tr(kspaceFirstOrder3D, simulation_options, execution_options)
    return np.asarray(p0_recon, dtype=np.float64)


def _reconstruct_time_reversal_pykwavers(
    *,
    pressure: np.ndarray,
    sensor_positions: np.ndarray,
    grid_points: tuple[int, int, int],
    spacing: tuple[float, float, float],
    sound_speed: float,
    dt: float,
    pml_size: tuple[int, int, int],
) -> np.ndarray:
    """Reconstruct the source image using the native pykwavers binding."""
    grid = kw.Grid(
        nx=int(grid_points[0]),
        ny=int(grid_points[1]),
        nz=int(grid_points[2]),
        dx=float(spacing[0]),
        dy=float(spacing[1]),
        dz=float(spacing[2]),
    )
    reconstruction = kw.time_reversal_reconstruction(
        np.asarray(pressure, dtype=np.float64),
        np.asarray(sensor_positions, dtype=np.float64),
        grid,
        float(sound_speed),
        float(1.0 / dt),
        pml_size=int(pml_size[0]),
    )
    return np.asarray(reconstruction, dtype=np.float64)


def run_kwave_reference() -> dict[str, np.ndarray | float]:
    """Run the k-wave-python reference simulation."""
    cached = _load_cached_result(KWAVE_CACHE)
    if cached is not None:
        return cached

    inputs = _build_example_inputs(scale=1)
    sound_speed = _sound_speed_scalar(inputs["medium"])
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
    """Run the comparison and return reconstruction metrics plus raw matrices."""
    inputs = _build_example_inputs(scale=1)
    source_pressure = np.asarray(inputs["source"].p0, dtype=np.float64)
    sound_speed = _sound_speed_scalar(inputs["medium"])
    grid_points = inputs["grid_points"]
    spacing = inputs["spacing"]
    pml_size = inputs["pml_size"]
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)
    sensor_positions = _mask_positions_fortran(sensor_mask, spacing)

    kw_results = run_kwave_reference()
    py_results = run_pykwavers_reference()

    kw_pressure_raw = np.asarray(kw_results["pressure"], dtype=np.float64)
    py_pressure_raw = np.asarray(py_results["pressure"], dtype=np.float64)
    kw_pressure = _align_sensor_matrix(kw_pressure_raw)
    py_pressure = _align_sensor_matrix(py_pressure_raw)
    if kw_pressure.shape != py_pressure.shape:
        raise AssertionError(f"sensor matrix shape mismatch: {kw_pressure.shape} != {py_pressure.shape}")

    kgrid = kWaveGrid(Vector(list(grid_points)), Vector(list(spacing)))
    kgrid.makeTime(sound_speed)

    kw_tr = kw_results.get("time_reversal")
    if kw_tr is None:
        kw_tr = _reconstruct_time_reversal_kwave(
            kgrid=kgrid,
            medium=inputs["medium"],
            sensor=inputs["sensor"],
            pressure=kw_pressure_raw,
            pml_size=pml_size,
        )
        kw_results = {**kw_results, "time_reversal": kw_tr}
        np.savez(KWAVE_CACHE, cache_version=CACHE_VERSION, **kw_results)

    py_tr = py_results.get("time_reversal")
    if py_tr is None:
        py_tr = _reconstruct_time_reversal_pykwavers(
            pressure=kw_pressure_raw,
            sensor_positions=sensor_positions,
            grid_points=grid_points,
            spacing=spacing,
            sound_speed=sound_speed,
            dt=float(kw_results["dt"]),
            pml_size=pml_size,
        )
        py_results = {**py_results, "time_reversal": py_tr}
        np.savez(PYKWAVERS_CACHE, cache_version=CACHE_VERSION, **py_results)

    if kw_tr.shape != py_tr.shape:
        raise AssertionError(f"time-reversal shape mismatch: {kw_tr.shape} != {py_tr.shape}")

    time_reversal_summary = compute_image_metrics(kw_tr, py_tr)
    reference_metrics = {
        "kwave_time_reversal": compute_image_metrics(source_pressure, kw_tr),
        "pykwavers_time_reversal": compute_image_metrics(source_pressure, py_tr),
    }

    representative_rows = [0, kw_pressure.shape[0] // 2, kw_pressure.shape[0] - 1]
    trace_metrics = {
        row: compute_trace_metrics(kw_pressure[row], py_pressure[row])
        for row in dict.fromkeys(representative_rows)
    }

    return {
        "kwave": {
            "pressure": kw_pressure,
            "time_reversal": kw_tr,
        },
        "pykwavers": {
            "pressure": py_pressure,
            "time_reversal": py_tr,
        },
        "summary": time_reversal_summary,
        "reference_metrics": reference_metrics,
        "trace_metrics": trace_metrics,
    }


_R_TARGET = 0.93
_RMS_MIN = 0.80
_RMS_MAX = 1.20
_PSNR_TARGET = 20.0


def main() -> int:
    """Execute the comparison, print metric summaries, and write a metrics file."""
    result = run_comparison()
    summary = result["summary"]

    r = float(summary["pearson_r"])
    rms = float(summary["rms_ratio"])
    psnr = float(summary["psnr_db"])
    checks = {
        "pearson_r": r >= _R_TARGET,
        "rms_ratio": _RMS_MIN <= rms <= _RMS_MAX,
        "psnr_db": psnr >= _PSNR_TARGET,
    }
    overall_status = "PASS" if all(checks.values()) else "FAIL"

    kw_ref = result["reference_metrics"]["kwave_time_reversal"]
    py_ref = result["reference_metrics"]["pykwavers_time_reversal"]

    print("=" * 80)
    print("k-wave-python pr_3D_TR_planar_sensor vs pykwavers")
    print("=" * 80)
    print(f"Time reversal Pearson r: {r:.6f}  (target >= {_R_TARGET})")
    print(f"Time reversal RMS ratio: {rms:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])")
    print(f"Time reversal PSNR [dB]: {psnr:.6f}  (target >= {_PSNR_TARGET})")
    print(f"k-Wave TR vs p0 r:       {kw_ref['pearson_r']:.6f}")
    print(f"pykwavers TR vs p0 r:    {py_ref['pearson_r']:.6f}")
    print(f"Status:                  {overall_status}")

    for row, metrics in result["trace_metrics"].items():
        print(
            f"Sensor row {row}: corr={metrics['pearson_r']:.6f}, "
            f"rmse={metrics['rmse']:.6e}, peak_ratio={metrics['peak_ratio']:.6f}"
        )

    # --- Structured metrics file ---
    output_path = DEFAULT_OUTPUT_DIR / "pr_3D_TR_planar_sensor_metrics.txt"
    with output_path.open("w") as fh:
        fh.write("pr_3D_TR_planar_sensor parity metrics\n")
        fh.write(f"parity_status: {overall_status}\n\n")
        fh.write("Time-reversal reconstruction (kwave vs pykwavers):\n")
        fh.write(f"  pearson_r = {r:.6f}  (target >= {_R_TARGET})\n")
        fh.write(f"  rms_ratio = {rms:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])\n")
        fh.write(f"  psnr_db   = {psnr:.6f}  (target >= {_PSNR_TARGET} dB)\n\n")
        fh.write("Reconstruction vs ground-truth p0:\n")
        fh.write(f"  kwave     pearson_r = {kw_ref['pearson_r']:.6f}\n")
        fh.write(f"  pykwavers pearson_r = {py_ref['pearson_r']:.6f}\n\n")
        fh.write("Forward sensor traces (representative rows):\n")
        for row, m in result["trace_metrics"].items():
            fh.write(
                f"  row={row}: pearson_r={m['pearson_r']:.6f}  "
                f"rms_ratio={m['rms_ratio']:.6f}  rmse={m['rmse']:.6e}\n"
            )
    print(f"Metrics written to: {output_path}")

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
