#!/usr/bin/env python3
"""
na_controlling_the_pml_compare.py
=================================
Parity comparison for the vendored upstream 2-D `na_controlling_the_PML`
example.

Contract
--------
The upstream MATLAB and Python examples define the same 2-D initial-pressure
problem:

* a 128x128 grid with 0.1 mm spacing
* two disc sources with the upstream positions and amplitudes
* a centered circular sensor mask with 50 Cartesian sample points
* four PML configurations:
  - `PMLAlpha = 0`
  - `PMLAlpha = 1e6`
  - `PMLSize = 2`
  - `PMLInside = False`

For each configuration the script runs k-wave-python and pykwavers with the
same physical domain, aligns the one-sample recorder phase offset, and compares
the full sensor matrices with value-based metrics. The parity gate is reported
per configuration and summarized across the whole sweep.

Outputs
-------
* `output/na_controlling_the_pml_compare.png`
* `output/na_controlling_the_pml_metrics.txt`
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    expand_pml_outside_shape,
    compute_image_metrics,
    pad_volume_for_pml_outside,
    normalize_sensor_matrix,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import smooth
from kwave.utils.conversion import cart2grid
from kwave.utils.mapgen import make_cart_circle, make_disc
from kwave_input_hdf5 import compare_kwave_input_files, write_kwave_input_file


NX = 128
NY = 128
DX = 0.1e-3
DY = 0.1e-3
C0 = 1500.0
RHO0 = 1000.0

SOURCE_1_MAG = 5.0
SOURCE_1_POS = Vector([50, 50])
SOURCE_1_RADIUS = 8

SOURCE_2_MAG = 3.0
SOURCE_2_POS = Vector([80, 60])
SOURCE_2_RADIUS = 5

SENSOR_RADIUS = 4e-3
NUM_SENSOR_POINTS = 50
PML_DEFAULT = 20

PARITY_THRESHOLDS = {
    "pearson_r": 0.99,
    "rms_ratio_min": 0.95,
    "rms_ratio_max": 1.05,
    "psnr_db": 30.0,
}

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "na_controlling_the_pml_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "na_controlling_the_pml_metrics.txt"
_CACHE_PREFIX = DEFAULT_OUTPUT_DIR / "na_controlling_the_pml"
CACHE_VERSION = 3
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
HDF5_PARITY_VERSION = 1
HDF5_OUTPUT_DIR = _CACHE_PREFIX / f"hdf5_v{HDF5_PARITY_VERSION}"

CONFIGS = (
    ("pml_alpha0", 0.0, None, True),
    ("pml_alpha1e6", 1.0e6, None, True),
    ("pml_size2", None, 2, True),
    ("pml_outside", None, None, False),
)


def _build_example_inputs() -> dict[str, object]:
    """Construct the shared 2-D k-Wave / pykwavers input state."""
    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    medium = kWaveMedium(sound_speed=C0, density=RHO0)
    kgrid.makeTime(medium.sound_speed)

    source_1 = SOURCE_1_MAG * make_disc(Vector([NX, NY]), SOURCE_1_POS, SOURCE_1_RADIUS)
    source_2 = SOURCE_2_MAG * make_disc(Vector([NX, NY]), SOURCE_2_POS, SOURCE_2_RADIUS)
    source_p0 = np.asarray(source_1 + source_2, dtype=np.float64)

    sensor_circle = make_cart_circle(SENSOR_RADIUS, NUM_SENSOR_POINTS, Vector([0, 0]))
    sensor_mask_grid, _, _ = cart2grid(kgrid, sensor_circle)
    sensor_mask = np.asarray(sensor_mask_grid, dtype=bool)
    sensor_count = int(np.count_nonzero(sensor_mask))
    if sensor_count == 0:
        raise AssertionError("Circular sensor mask has no active points")

    return {
        "kgrid": kgrid,
        "medium": medium,
        "source_p0": source_p0,
        "sensor_mask": sensor_mask,
        "sensor_count": sensor_count,
    }


def _align_sensor_matrix(pressure: np.ndarray, expected_sensors: int) -> np.ndarray:
    """Normalize the matrix shape and remove the recorder phase offset."""
    aligned = normalize_sensor_matrix(np.asarray(pressure, dtype=np.float64), expected_sensors=expected_sensors)
    if aligned.ndim != 2:
        raise AssertionError(f"expected a 2-D sensor matrix, got {aligned.shape}")
    if aligned.shape[1] > 1:
        aligned = aligned[:, 1:]
    return aligned


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
        "runtime_s": float(cached["runtime_s"]),
        "sensor_count": float(cached["sensor_count"]),
    }


def _prepare_pykwavers_case_state(
    *,
    pml_alpha: float | None,
    pml_size: int | None,
    pml_inside: bool,
) -> dict[str, object]:
    inputs = _build_example_inputs()
    source_p0 = np.asarray(inputs["source_p0"], dtype=np.float64)
    source_p0 = np.asarray(smooth(source_p0, True), dtype=np.float64)
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)
    kw_pml_size = PML_DEFAULT if pml_size is None else int(pml_size)

    if pml_inside:
        grid = pkw.Grid(nx=NX, ny=NY, nz=1, dx=DX, dy=DY, dz=DX)
        source_p0_py = source_p0
        sensor_mask_py = sensor_mask[:, :, None]
        sim_pml_inside = True
    else:
        expanded_grid = expand_pml_outside_shape((NX, NY), (kw_pml_size, kw_pml_size))
        grid = pkw.Grid(
            nx=int(expanded_grid[0]),
            ny=int(expanded_grid[1]),
            nz=int(expanded_grid[2]),
            dx=DX,
            dy=DY,
            dz=DX,
        )
        source_p0_py = pad_volume_for_pml_outside(source_p0, (kw_pml_size, kw_pml_size))
        sensor_mask_py = pad_volume_for_pml_outside(sensor_mask, (kw_pml_size, kw_pml_size)).astype(bool)
        sim_pml_inside = True

    return {
        "inputs": inputs,
        "grid": grid,
        "source_p0": source_p0_py,
        "sensor_mask": sensor_mask_py,
        "sensor_count": int(inputs["sensor_count"]),
        "kw_pml_size": kw_pml_size,
        "pml_alpha": 2.0 if pml_alpha is None else float(pml_alpha),
        "pml_inside": sim_pml_inside,
    }


def _prepare_hdf5_paths(config_name: str) -> tuple[Path, Path]:
    HDF5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_path = HDF5_OUTPUT_DIR / f"kwave_{config_name}_v{HDF5_PARITY_VERSION}.h5"
    candidate_path = HDF5_OUTPUT_DIR / f"pykwavers_{config_name}_v{HDF5_PARITY_VERSION}.h5"
    return reference_path, candidate_path


def _write_reference_input_only(
    *,
    config_name: str,
    pml_alpha: float | None,
    pml_size: int | None,
    pml_inside: bool,
) -> Path:
    inputs = _build_example_inputs()
    kgrid = inputs["kgrid"]
    medium = inputs["medium"]
    source_p0 = np.asarray(inputs["source_p0"], dtype=np.float64)
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)

    source = kSource()
    source.p0 = source_p0

    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]

    kw_pml_size = PML_DEFAULT if pml_size is None else int(pml_size)
    kw_pml_alpha = 2.0 if pml_alpha is None else float(pml_alpha)
    simulation_options = SimulationOptions(
        pml_inside=pml_inside,
        pml_size=kw_pml_size,
        pml_alpha=kw_pml_alpha,
        smooth_p0=True,
        data_cast="single",
        save_to_disk=True,
        save_to_disk_exit=True,
        input_filename=f"na_controlling_the_pml_{config_name}.h5",
        data_path=tempfile.gettempdir(),
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        verbose_level=0,
        show_sim_log=False,
    )

    print(f"  [k-wave/{config_name}] Writing reference input file...")
    start = time.perf_counter()
    kspaceFirstOrder2D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    runtime_s = time.perf_counter() - start
    print(f"  [k-wave/{config_name}] Reference input written in {runtime_s:.1f} s")

    temp_path = Path(tempfile.gettempdir()) / f"na_controlling_the_pml_{config_name}.h5"
    if not temp_path.exists():
        raise FileNotFoundError(f"Expected reference input file was not written: {temp_path}")

    reference_path, _ = _prepare_hdf5_paths(config_name)
    shutil.copy2(temp_path, reference_path)
    return reference_path


def _compare_hdf5_case(
    *,
    config_name: str,
    pml_alpha: float | None,
    pml_size: int | None,
    pml_inside: bool,
    no_cache: bool,
) -> dict[str, object]:
    reference_path, candidate_path = _prepare_hdf5_paths(config_name)
    if REFRESH_CACHE or no_cache or not reference_path.exists():
        reference_path = _write_reference_input_only(
            config_name=config_name,
            pml_alpha=pml_alpha,
            pml_size=pml_size,
            pml_inside=pml_inside,
        )

    state = _prepare_pykwavers_case_state(
        pml_alpha=pml_alpha,
        pml_size=pml_size,
        pml_inside=pml_inside,
    )

    inputs = state["inputs"]
    grid = state["grid"]
    kgrid = inputs["kgrid"]
    source_p0 = np.asarray(state["source_p0"], dtype=np.float64)
    sensor_mask = np.asarray(state["sensor_mask"], dtype=bool)
    root_attrs = {}
    with h5py.File(reference_path, "r") as ref_h5:
        for key in ("created_by", "creation_date", "file_description", "file_type", "major_version", "minor_version"):
            root_attrs[key] = ref_h5.attrs[key]

    write_kwave_input_file(
        candidate_path,
        grid_shape=(int(grid.nx), int(grid.ny), int(grid.nz)),
        grid_spacing=(float(grid.dx), float(grid.dy), float(grid.dz)),
        nt=int(kgrid.Nt),
        dt=float(kgrid.dt),
        pml_size=(int(state["kw_pml_size"]), int(state["kw_pml_size"]), 0),
        pml_alpha=(float(state["pml_alpha"]), float(state["pml_alpha"]), 0.0),
        c0=C0,
        c_ref=C0,
        rho0=RHO0,
        source_p0=source_p0,
        sensor_mask=sensor_mask,
        root_attrs=root_attrs,
    )

    comparison = compare_kwave_input_files(reference_path, candidate_path)
    comparison["config_name"] = config_name
    comparison["reference_path"] = str(reference_path)
    comparison["candidate_path"] = str(candidate_path)
    comparison["sensor_count"] = float(state["sensor_count"])
    return comparison


def _run_kwave_case(
    *,
    config_name: str,
    pml_alpha: float | None,
    pml_size: int | None,
    pml_inside: bool,
    no_cache: bool,
) -> dict[str, np.ndarray | float]:
    cache = _CACHE_PREFIX / f"kwave_{config_name}.npz"
    if not no_cache:
        cached = _load_cached_result(cache)
        if cached is not None:
            return cached

    inputs = _build_example_inputs()
    kgrid = inputs["kgrid"]
    medium = inputs["medium"]
    source_p0 = np.asarray(inputs["source_p0"], dtype=np.float64)
    sensor_mask = np.asarray(inputs["sensor_mask"], dtype=bool)
    sensor_count = int(inputs["sensor_count"])

    source = kSource()
    source.p0 = source_p0

    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]

    kw_pml_size = PML_DEFAULT if pml_size is None else int(pml_size)
    kw_pml_alpha = 2.0 if pml_alpha is None else float(pml_alpha)
    simulation_options = SimulationOptions(
        pml_inside=pml_inside,
        pml_size=kw_pml_size,
        pml_alpha=kw_pml_alpha,
        smooth_p0=True,
        data_cast="single",
        save_to_disk=True,
        input_filename=f"na_controlling_the_pml_{config_name}.h5",
        data_path=tempfile.gettempdir(),
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        verbose_level=0,
        show_sim_log=False,
    )

    print(f"  [k-wave/{config_name}] Running...")
    start = time.perf_counter()
    sensor_data = kspaceFirstOrder2D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    runtime_s = time.perf_counter() - start
    print(f"  [k-wave/{config_name}] Done in {runtime_s:.1f} s")

    pressure = normalize_sensor_matrix(np.asarray(sensor_data["p"], dtype=np.float64), expected_sensors=sensor_count)
    output = {
        "pressure": pressure,
        "time": np.asarray(kgrid.t_array, dtype=np.float64).ravel(),
        "dt": float(kgrid.dt),
        "runtime_s": runtime_s,
        "sensor_count": float(sensor_count),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, cache_version=CACHE_VERSION, **output)
    return output


def _run_pykwavers_case(
    *,
    config_name: str,
    pml_alpha: float | None,
    pml_size: int | None,
    pml_inside: bool,
    no_cache: bool,
) -> dict[str, np.ndarray | float]:
    cache = _CACHE_PREFIX / f"pykwavers_{config_name}.npz"
    if not no_cache:
        cached = _load_cached_result(cache)
        if cached is not None:
            return cached

    state = _prepare_pykwavers_case_state(
        pml_alpha=pml_alpha,
        pml_size=pml_size,
        pml_inside=pml_inside,
    )
    inputs = state["inputs"]
    kgrid = inputs["kgrid"]
    grid = state["grid"]
    source_p0_py = np.asarray(state["source_p0"], dtype=np.float64)
    sensor_mask_py = np.asarray(state["sensor_mask"], dtype=bool)
    sensor_count = int(state["sensor_count"])
    kw_pml_size = int(state["kw_pml_size"])
    sim_pml_inside = bool(state["pml_inside"])

    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_initial_pressure(source_p0_py)
    sensor = pkw.Sensor.from_mask(sensor_mask_py)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(kw_pml_size)
    sim.set_pml_inside(sim_pml_inside)
    sim.set_pml_alpha(2.0 if pml_alpha is None else float(pml_alpha))

    print(f"  [pykwavers/{config_name}] Running...")
    start = time.perf_counter()
    result = sim.run(time_steps=int(kgrid.Nt), dt=float(kgrid.dt))
    runtime_s = time.perf_counter() - start
    print(f"  [pykwavers/{config_name}] Done in {runtime_s:.1f} s")

    pressure = normalize_sensor_matrix(np.asarray(result.sensor_data, dtype=np.float64), expected_sensors=sensor_count)
    output = {
        "pressure": pressure,
        "time": np.asarray(result.time, dtype=np.float64).ravel(),
        "dt": float(result.dt),
        "runtime_s": runtime_s,
        "sensor_count": float(sensor_count),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, cache_version=CACHE_VERSION, **output)
    return output


def _run_hdf5_sweep(*, no_cache: bool) -> dict[str, object]:
    case_results: dict[str, dict[str, object]] = {}
    summary = {
        "dataset_max_abs_diff_max": 0.0,
        "root_attr_mismatch_count": 0,
        "status": "PASS",
    }

    for config_name, pml_alpha, pml_size, pml_inside in CONFIGS:
        comparison = _compare_hdf5_case(
            config_name=config_name,
            pml_alpha=pml_alpha,
            pml_size=pml_size,
            pml_inside=pml_inside,
            no_cache=no_cache,
        )
        case_results[config_name] = comparison
        summary["dataset_max_abs_diff_max"] = max(summary["dataset_max_abs_diff_max"], float(comparison["max_abs_diff"]))
        summary["root_attr_mismatch_count"] += len(comparison["root_attr_mismatches"])
        if comparison["status"] != "PASS":
            summary["status"] = "FAIL"

    return {
        "cases": case_results,
        "summary": summary,
        "status": summary["status"],
    }


def _plot_case(ax_kw, ax_py, ax_diff, *, kw_pressure: np.ndarray, py_pressure: np.ndarray, case_title: str) -> None:
    vmax = float(max(np.max(np.abs(kw_pressure)), np.max(np.abs(py_pressure)), 1.0e-30))
    diff = py_pressure - kw_pressure
    dmax = float(max(np.max(np.abs(diff)), 1.0e-30))

    im_kw = ax_kw.imshow(
        kw_pressure,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    im_py = ax_py.imshow(
        py_pressure,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    im_diff = ax_diff.imshow(
        diff,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        vmin=-dmax,
        vmax=dmax,
    )

    ax_kw.set_title(f"{case_title} k-Wave", fontsize=8)
    ax_py.set_title(f"{case_title} pykwavers", fontsize=8)
    ax_diff.set_title(f"{case_title} diff", fontsize=8)

    ax_kw.set_ylabel("Sensor")
    ax_py.set_ylabel("Sensor")
    ax_diff.set_ylabel("Sensor")

    for ax in (ax_kw, ax_py, ax_diff):
        ax.set_xlabel("Time step")

    return im_kw, im_py, im_diff


def _run_sweep(*, no_cache: bool) -> dict[str, object]:
    """Run the full PML sweep and return aligned matrices plus metrics."""
    case_results: dict[str, dict[str, object]] = {}
    summary = {
        "pearson_r_min": float("inf"),
        "rms_ratio_min": float("inf"),
        "rms_ratio_max": 0.0,
        "psnr_db_min": float("inf"),
        "max_abs_diff_max": 0.0,
    }

    for config_name, pml_alpha, pml_size, pml_inside in CONFIGS:
        kw = _run_kwave_case(
            config_name=config_name,
            pml_alpha=pml_alpha,
            pml_size=pml_size,
            pml_inside=pml_inside,
            no_cache=no_cache,
        )
        py = _run_pykwavers_case(
            config_name=config_name,
            pml_alpha=pml_alpha,
            pml_size=pml_size,
            pml_inside=pml_inside,
            no_cache=no_cache,
        )

        kw_pressure = _align_sensor_matrix(kw["pressure"], expected_sensors=int(kw["sensor_count"]))
        py_pressure = _align_sensor_matrix(py["pressure"], expected_sensors=int(py["sensor_count"]))
        if kw_pressure.shape != py_pressure.shape:
            raise AssertionError(
                f"{config_name}: sensor matrix shape mismatch {kw_pressure.shape} != {py_pressure.shape}"
            )

        metrics = compute_image_metrics(kw_pressure, py_pressure)
        case_results[config_name] = {
            "kwave": {**kw, "pressure": kw_pressure},
            "pykwavers": {**py, "pressure": py_pressure},
            "metrics": metrics,
            "pml_alpha": pml_alpha,
            "pml_size": pml_size,
            "pml_inside": pml_inside,
        }

        summary["pearson_r_min"] = min(summary["pearson_r_min"], metrics["pearson_r"])
        summary["rms_ratio_min"] = min(summary["rms_ratio_min"], metrics["rms_ratio"])
        summary["rms_ratio_max"] = max(summary["rms_ratio_max"], metrics["rms_ratio"])
        summary["psnr_db_min"] = min(summary["psnr_db_min"], metrics["psnr_db"])
        summary["max_abs_diff_max"] = max(summary["max_abs_diff_max"], metrics["max_abs_diff"])

    status = "PASS" if (
        summary["pearson_r_min"] >= PARITY_THRESHOLDS["pearson_r"]
        and summary["rms_ratio_min"] >= PARITY_THRESHOLDS["rms_ratio_min"]
        and summary["rms_ratio_max"] <= PARITY_THRESHOLDS["rms_ratio_max"]
        and summary["psnr_db_min"] >= PARITY_THRESHOLDS["psnr_db"]
    ) else "FAIL"

    return {
        "cases": case_results,
        "summary": summary,
        "status": status,
    }


def run_comparison(*, no_cache: bool = False) -> dict[str, object]:
    """Run the full PML sweep and the exact save-to-disk HDF5 parity sweep."""
    waveform = _run_sweep(no_cache=no_cache)
    hdf5 = _run_hdf5_sweep(no_cache=no_cache)
    status = "PASS" if waveform["status"] == "PASS" and hdf5["status"] == "PASS" else "FAIL"
    return {
        "waveform": waveform,
        "hdf5": hdf5,
        "status": status,
        "cases": {
            name: {
                **waveform["cases"][name],
                "hdf5": hdf5["cases"][name],
            }
            for name, *_ in CONFIGS
        },
        "summary": waveform["summary"],
        "hdf5_summary": hdf5["summary"],
    }


def plot_comparison(results: dict[str, object]) -> None:
    """Render the k-Wave and pykwavers sensor matrices for each PML case."""
    fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(16, 14), sharex=True, sharey=True)
    if len(CONFIGS) == 1:
        axes = np.asarray([axes], dtype=object)

    fig.suptitle(
        "na_controlling_the_pml: k-wave-python vs pykwavers\n"
        f"Grid {NX}x{NY}, dx={DX*1e3:.1f} mm, sensor points={NUM_SENSOR_POINTS}",
        fontsize=11,
    )

    for row, (config_name, _, _, _) in enumerate(CONFIGS):
        case = results["cases"][config_name]
        kw_pressure = np.asarray(case["kwave"]["pressure"], dtype=np.float64)
        py_pressure = np.asarray(case["pykwavers"]["pressure"], dtype=np.float64)
        _plot_case(
            axes[row, 0],
            axes[row, 1],
            axes[row, 2],
            kw_pressure=kw_pressure,
            py_pressure=py_pressure,
            case_title=config_name,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(str(FIGURE_PATH), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("na_controlling_the_pml")
    print(f"  Grid   : {NX}x{NY}  dx={DX*1e3:.2f} mm")
    print(f"  Sensor : {NUM_SENSOR_POINTS} circular points")
    print("=" * 72)

    results = run_comparison(no_cache=args.no_cache)
    waveform = results["waveform"]
    hdf5 = results["hdf5"]
    summary = waveform["summary"]
    status = results["status"]

    print("\n  --- parity summary ---")
    print(f"    pearson_r_min : {summary['pearson_r_min']:.6f}")
    print(f"    rms_ratio_min : {summary['rms_ratio_min']:.6f}")
    print(f"    rms_ratio_max : {summary['rms_ratio_max']:.6f}")
    print(f"    psnr_db_min   : {summary['psnr_db_min']:.6f}")
    print(f"    max_abs_diff  : {summary['max_abs_diff_max']:.6e}")
    print(f"    waveform_status : {waveform['status']}")
    print(f"    hdf5_status     : {hdf5['status']}")
    print(f"    status          : {status}")

    for config_name, _, _, _ in CONFIGS:
        metrics = results["cases"][config_name]["metrics"]
        hdf5_metrics = results["cases"][config_name]["hdf5"]
        print(
            f"    {config_name}: r={metrics['pearson_r']:.6f}, "
            f"rms={metrics['rms_ratio']:.6f}, psnr={metrics['psnr_db']:.3f}, "
            f"runtime_kw={results['cases'][config_name]['kwave']['runtime_s']:.1f}s, "
            f"runtime_py={results['cases'][config_name]['pykwavers']['runtime_s']:.1f}s, "
            f"hdf5={hdf5_metrics['status']}"
        )

    plot_comparison(waveform)

    lines = [
        f"grid: {NX}x{NY}  dx={DX*1e3:.3f}mm  sensor_points={NUM_SENSOR_POINTS}",
        "",
        "parity summary",
        "--------------",
        f"  pearson_r_min : {summary['pearson_r_min']}",
        f"  rms_ratio_min : {summary['rms_ratio_min']}",
        f"  rms_ratio_max : {summary['rms_ratio_max']}",
        f"  psnr_db_min   : {summary['psnr_db_min']}",
        f"  max_abs_diff  : {summary['max_abs_diff_max']}",
        f"  waveform_status : {waveform['status']}",
        f"  hdf5_status     : {hdf5['status']}",
        f"  status          : {status}",
        "",
    ]
    for config_name, _, _, _ in CONFIGS:
        case = results["cases"][config_name]
        metrics = case["metrics"]
        hdf5_metrics = case["hdf5"]
        lines.extend(
            [
                f"config: {config_name}",
                f"  pml_alpha  = {case['pml_alpha']}",
                f"  pml_size   = {case['pml_size']}",
                f"  pml_inside = {case['pml_inside']}",
                f"  pearson_r  = {metrics['pearson_r']}",
                f"  rms_ratio  = {metrics['rms_ratio']}",
                f"  rmse       = {metrics['rmse']}",
                f"  max_abs_diff = {metrics['max_abs_diff']}",
                f"  peak_ratio = {metrics['peak_ratio']}",
                f"  psnr_db    = {metrics['psnr_db']}",
                f"  runtime_kw = {case['kwave']['runtime_s']}",
                f"  runtime_py = {case['pykwavers']['runtime_s']}",
                f"  hdf5_status = {hdf5_metrics['status']}",
                f"  hdf5_max_abs_diff = {hdf5_metrics['max_abs_diff']}",
                f"  hdf5_reference = {hdf5_metrics['reference_path']}",
                f"  hdf5_candidate = {hdf5_metrics['candidate_path']}",
                "",
            ]
        )

    save_text_report(METRICS_PATH, "na_controlling_the_pml", lines)

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
