#!/usr/bin/env python3
"""
at_circular_piston_3D_compare.py
===============================
Side-by-side comparison of k-wave-python vs pykwavers for the circular piston
transducer example.  Both engines are driven with the same continuous-wave
signal and the on-axis steady-state pressure profile is compared against the
analytical infinite-baffle piston solution from Pierce.

The script keeps the physical configuration consistent with the vendored
k-wave-python example while using the native pykwavers `KWaveArray` disc
geometry on the full grid and an outer PML padding layer so the boundary
treatment remains explicit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    clip_volume_to_physical_interior,
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_trace_metrics,
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
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import extract_amp_phase
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.math import round_even
from kwave.utils.signals import create_cw_signals


AXIAL_SIZE = 32e-3
LATERAL_SIZE = 23e-3
SOURCE_F0 = 1.0e6
SOURCE_DIAMETER = 10e-3
SOURCE_AMP = 1.0e6
C0 = 1500.0
RHO0 = 1000.0
PPW = 3
CFL = 0.5
PML_SIZE = 10
RECORD_PERIODS = 1

DX = C0 / (PPW * SOURCE_F0)
PPP = int(round(PPW / CFL))
DT = 1.0 / (PPP * SOURCE_F0)
T_END = 40e-6
NT = int(round(T_END / DT))

TOTAL_NX = round_even(AXIAL_SIZE / DX)
TOTAL_NY = round_even(LATERAL_SIZE / DX)
TOTAL_NZ = TOTAL_NY

PHYSICAL_NX = TOTAL_NX - 2 * PML_SIZE
PHYSICAL_NY = TOTAL_NY - 2 * PML_SIZE
PHYSICAL_NZ = TOTAL_NZ - 2 * PML_SIZE

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "at_circular_piston_3D_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "at_circular_piston_3D_metrics.txt"

KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "at_circular_piston_3D_kwave_cache.npz"
PKWAVERS_CACHE = DEFAULT_OUTPUT_DIR / "at_circular_piston_3D_pykwavers_cache.npz"
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"

SOURCE_CENTER = (PML_SIZE * DX, TOTAL_NY * DX / 2.0, TOTAL_NZ * DX / 2.0)
SOURCE_FOCUS_DISTANCE = 1.0e-3
SOURCE_FOCUS = (
    SOURCE_CENTER[0] + SOURCE_FOCUS_DISTANCE,
    SOURCE_CENTER[1],
    SOURCE_CENTER[2],
)


def build_signal() -> np.ndarray:
    """Return the shared CW drive signal."""
    t = np.arange(NT) * DT
    sig = create_cw_signals(t, SOURCE_F0, np.array([SOURCE_AMP]), np.array([0.0]))
    return np.asarray(sig, dtype=np.float64).flatten()


def _load_cached_result(path: os.PathLike[str] | str) -> dict[str, np.ndarray | float] | None:
    if REFRESH_CACHE:
        return None
    cache_path = os.fspath(path)
    if not os.path.exists(cache_path):
        return None
    cached = np.load(cache_path, allow_pickle=False)
    return {
        "pressure": np.asarray(cached["pressure"], dtype=np.float64),
        "amp_on_axis": np.asarray(cached["amp_on_axis"], dtype=np.float64),
        "time": np.asarray(cached["time"], dtype=np.float64),
        "dt": float(cached["dt"]),
        "runtime_s": float(cached["runtime_s"]),
    }


def _build_piston_sensor_mask(nx: int, ny: int, nz: int) -> np.ndarray:
    """Build the on-axis line sensor mask used by the example."""
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[1:, :, nz // 2] = True
    return mask


def _normalize_sensor_data(data: np.ndarray, expected_sensors: int) -> np.ndarray:
    """Return sensor data as `(n_sensors, n_time_samples)`."""
    return normalize_sensor_matrix(np.asarray(data, dtype=np.float64), expected_sensors=expected_sensors)


def _extract_steady_state_amplitude(
    pressure: np.ndarray,
    *,
    dt: float,
    sensor_count: int,
) -> np.ndarray:
    """Extract the axial steady-state amplitude profile from sensor traces."""
    pressure = _normalize_sensor_data(pressure, expected_sensors=sensor_count)
    pressure = pressure[:, -RECORD_PERIODS * PPP :]
    amp, _, _ = extract_amp_phase(
        pressure,
        1.0 / dt,
        SOURCE_F0,
        dim=1,
        fft_padding=1,
        window="Rectangular",
    )
    amp = np.asarray(amp, dtype=np.float64).flatten()
    amp_plane = _reshape_axial_amplitude(amp)
    return amp_plane[:, PHYSICAL_NY // 2]


def _reshape_axial_amplitude(amp: np.ndarray) -> np.ndarray:
    """Reshape the flattened amplitude vector to the axial/lateral grid."""
    return np.reshape(np.asarray(amp, dtype=np.float64), (PHYSICAL_NX - 1, PHYSICAL_NY), order="F")


def _analytical_piston_profile() -> tuple[np.ndarray, np.ndarray]:
    """Return the exact infinite-baffle axial pressure for the piston example."""
    k = 2.0 * np.pi * SOURCE_F0 / C0
    a = SOURCE_DIAMETER / 2.0
    x_vec = np.arange(1, PHYSICAL_NX, dtype=np.float64) * DX
    r = np.sqrt(x_vec**2 + a**2)
    p_ref = SOURCE_AMP * np.abs(2.0 * np.sin((k * r - k * x_vec) / 2.0))
    return x_vec, p_ref


def _build_kwave_configuration() -> tuple[kWaveGrid, kWaveMedium, kSource, kSensor, np.ndarray, np.ndarray]:
    """Build the canonical k-wave-python configuration."""
    kgrid = kWaveGrid(Vector([PHYSICAL_NX, PHYSICAL_NY, PHYSICAL_NZ]), Vector([DX, DX, DX]))
    kgrid.setTime(NT, DT)

    medium = kWaveMedium(sound_speed=C0, density=RHO0)

    input_signal = build_signal()
    karray = KWaveArray_Kwave(bli_tolerance=0.03, upsampling_rate=10, single_precision=True)
    karray.add_disc_element([kgrid.x_vec[0].item(), 0.0, 0.0], SOURCE_DIAMETER, [0.0, 0.0, 0.0])
    source_weights = np.asarray(karray.get_array_grid_weights(kgrid), dtype=np.float64)

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, input_signal.reshape(1, -1))

    sensor = kSensor()
    sensor.mask = _build_piston_sensor_mask(PHYSICAL_NX, PHYSICAL_NY, PHYSICAL_NZ)
    sensor.record = ["p"]
    sensor.record_start_index = kgrid.Nt - (RECORD_PERIODS * PPP) + 1

    return kgrid, medium, source, sensor, input_signal, source_weights


def _build_pykwavers_configuration(
    input_signal: np.ndarray,
) -> tuple[pkw.Grid, pkw.Medium, pkw.Source, pkw.Sensor, np.ndarray]:
    """Build the matching pykwavers configuration."""
    grid = pkw.Grid(
        nx=TOTAL_NX,
        ny=TOTAL_NY,
        nz=TOTAL_NZ,
        dx=DX,
        dy=DX,
        dz=DX,
    )
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    karray = pkw.KWaveArray()
    karray.set_sound_speed(C0)
    karray.set_frequency(SOURCE_F0)
    karray.add_disc_element(SOURCE_CENTER, SOURCE_DIAMETER, SOURCE_FOCUS)
    source_weights = np.asarray(karray.get_array_weighted_mask(grid), dtype=np.float64)
    source = pkw.Source.from_kwave_array(karray, input_signal, SOURCE_F0, mode="additive")

    sensor_mask = _build_piston_sensor_mask(PHYSICAL_NX, PHYSICAL_NY, PHYSICAL_NZ)
    sensor_mask = pad_volume_for_pml_outside(sensor_mask.astype(bool), (PML_SIZE, PML_SIZE, PML_SIZE)).astype(bool)
    sensor = pkw.Sensor.from_mask(sensor_mask)

    return grid, medium, source, sensor, source_weights


def run_kwave_reference() -> dict[str, np.ndarray | float]:
    """Run the k-wave-python reference simulation."""
    cached = _load_cached_result(KWAVE_CACHE)
    kgrid, medium, source, sensor, input_signal, source_weights = _build_kwave_configuration()
    if cached is not None:
        return {**cached, "input_signal": np.asarray(input_signal, dtype=np.float64), "source_weights": source_weights}

    sim_opts = SimulationOptions(
        pml_inside=False,
        pml_size=Vector([PML_SIZE, PML_SIZE, PML_SIZE]),
        data_cast="single",
        save_to_disk=True,
    )
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

    start = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    runtime = time.perf_counter() - start

    pressure = _normalize_sensor_data(sensor_data["p"], expected_sensors=(PHYSICAL_NX - 1) * PHYSICAL_NY)
    amp_on_axis = _extract_steady_state_amplitude(pressure, dt=DT, sensor_count=(PHYSICAL_NX - 1) * PHYSICAL_NY)
    output = {
        "pressure": pressure,
        "amp_on_axis": amp_on_axis,
        "time": np.asarray(kgrid.t_array, dtype=np.float64),
        "dt": float(kgrid.dt),
        "runtime_s": runtime,
        "input_signal": np.asarray(input_signal, dtype=np.float64),
        "source_weights": source_weights,
    }
    np.savez(
        KWAVE_CACHE,
        pressure=output["pressure"],
        amp_on_axis=output["amp_on_axis"],
        time=output["time"],
        dt=output["dt"],
        runtime_s=output["runtime_s"],
    )
    return output


def run_pykwavers(input_signal: np.ndarray) -> dict[str, np.ndarray | float]:
    """Run the pykwavers counterpart simulation."""
    cached = _load_cached_result(PKWAVERS_CACHE)
    grid, medium, source, sensor, source_weights = _build_pykwavers_configuration(input_signal)
    if cached is not None:
        return {**cached, "source_weights": source_weights}

    n_steps = NT
    start = time.perf_counter()
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)
    result = sim.run(time_steps=n_steps, dt=DT)
    runtime = time.perf_counter() - start

    pressure = _normalize_sensor_data(result.sensor_data, expected_sensors=(PHYSICAL_NX - 1) * PHYSICAL_NY)
    amp_on_axis = _extract_steady_state_amplitude(pressure, dt=DT, sensor_count=(PHYSICAL_NX - 1) * PHYSICAL_NY)
    output = {
        "pressure": pressure,
        "amp_on_axis": amp_on_axis,
        "time": np.asarray(result.time, dtype=np.float64),
        "dt": float(result.dt),
        "runtime_s": runtime,
        "source_weights": source_weights,
    }
    np.savez(
        PKWAVERS_CACHE,
        pressure=output["pressure"],
        amp_on_axis=output["amp_on_axis"],
        time=output["time"],
        dt=output["dt"],
        runtime_s=output["runtime_s"],
    )
    return output


def plot_comparison(kwave: dict[str, np.ndarray | float], pykwavers: dict[str, np.ndarray | float], x_vec: np.ndarray, p_ref: np.ndarray) -> None:
    """Save a comparison figure for the on-axis amplitude profile."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_vec * 1e3, p_ref * 1e-6, "k-", linewidth=1.8, alpha=0.85, label="Analytical")
    ax.plot(x_vec * 1e3, np.asarray(kwave["amp_on_axis"], dtype=np.float64) * 1e-6, "b.", label="k-wave-python")
    ax.plot(
        x_vec * 1e3,
        np.asarray(pykwavers["amp_on_axis"], dtype=np.float64) * 1e-6,
        "r--",
        linewidth=1.2,
        label="pykwavers (native KWaveArray)",
    )
    ax.axvline(SOURCE_DIAMETER * 0.5 * 1e3, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Axial distance from piston [mm]")
    ax.set_ylabel("Steady-state pressure amplitude [MPa]")
    ax.set_title(
        "at_circular_piston_3D: k-wave-python vs pykwavers\n"
        f"diameter={SOURCE_DIAMETER * 1e3:.0f} mm, f0={SOURCE_F0 * 1e-6:.1f} MHz, "
        f"grid {PHYSICAL_NX}x{PHYSICAL_NY}x{PHYSICAL_NZ} (physical)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_report_lines(
    kwave: dict[str, np.ndarray | float],
    pykwavers: dict[str, np.ndarray | float],
    analytical: np.ndarray,
    source_metrics: dict[str, float],
    metrics_kwave: dict[str, float],
    metrics_pykwavers: dict[str, float],
    metrics_between: dict[str, float],
) -> list[str]:
    """Build the plain-text metrics report."""
    return [
        "example: at_circular_piston_3D",
        f"physical_grid: {PHYSICAL_NX}x{PHYSICAL_NY}x{PHYSICAL_NZ}",
        f"total_grid: {TOTAL_NX}x{TOTAL_NY}x{TOTAL_NZ}",
        f"dt_s: {DT:.9e}",
        f"kwave_runtime_s: {float(kwave['runtime_s']):.3f}",
        f"pykwavers_runtime_s: {float(pykwavers['runtime_s']):.3f}",
        "",
        "source weights (physical interior): k-wave-python vs pykwavers",
        f"  pearson_r = {source_metrics['pearson_r']:.6f}",
        f"  rms_ratio = {source_metrics['rms_ratio']:.6f}",
        f"  rmse      = {source_metrics['rmse']:.6e}",
        f"  peak_ratio= {source_metrics['peak_ratio']:.6f}",
        "",
        "k-wave-python vs analytical:",
        f"  pearson_r = {metrics_kwave['pearson_r']:.6f}",
        f"  rms_ratio = {metrics_kwave['rms_ratio']:.6f}",
        f"  rmse      = {metrics_kwave['rmse']:.6e}",
        f"  peak_ratio= {metrics_kwave['peak_ratio']:.6f}",
        "",
        "pykwavers vs analytical:",
        f"  pearson_r = {metrics_pykwavers['pearson_r']:.6f}",
        f"  rms_ratio = {metrics_pykwavers['rms_ratio']:.6f}",
        f"  rmse      = {metrics_pykwavers['rmse']:.6e}",
        f"  peak_ratio= {metrics_pykwavers['peak_ratio']:.6f}",
        "",
        "k-wave-python vs pykwavers:",
        f"  pearson_r = {metrics_between['pearson_r']:.6f}",
        f"  rms_ratio = {metrics_between['rms_ratio']:.6f}",
        f"  rmse      = {metrics_between['rmse']:.6e}",
        f"  peak_ratio= {metrics_between['peak_ratio']:.6f}",
        "",
        f"analytical_peak_pa = {float(np.max(np.asarray(analytical, dtype=np.float64))):.6e}",
        f"kwave_peak_pa      = {float(np.max(np.asarray(kwave['amp_on_axis'], dtype=np.float64))):.6e}",
        f"pykwavers_peak_pa  = {float(np.max(np.asarray(pykwavers['amp_on_axis'], dtype=np.float64))):.6e}",
    ]


def run_comparison() -> dict[str, object]:
    """Run the comparison and return amplitudes plus metrics."""
    input_signal = build_signal()
    kwave = run_kwave_reference()
    pykwavers = run_pykwavers(input_signal)
    x_vec, analytical = _analytical_piston_profile()

    kw_amp = np.asarray(kwave["amp_on_axis"], dtype=np.float64)
    py_amp = np.asarray(pykwavers["amp_on_axis"], dtype=np.float64)
    kw_source_weights = pad_volume_for_pml_outside(
        np.asarray(kwave["source_weights"], dtype=np.float64),
        (PML_SIZE, PML_SIZE, PML_SIZE),
    )
    py_source_weights = clip_volume_to_physical_interior(
        np.asarray(pykwavers["source_weights"], dtype=np.float64),
        (PML_SIZE, PML_SIZE, PML_SIZE),
    )
    if kw_amp.shape != py_amp.shape:
        raise AssertionError(f"on-axis amplitude shape mismatch: {kw_amp.shape} != {py_amp.shape}")
    if kw_source_weights.shape != py_source_weights.shape:
        raise AssertionError(
            f"source weight shape mismatch: {kw_source_weights.shape} != {py_source_weights.shape}"
        )

    metrics_kwave = compute_trace_metrics(analytical, kw_amp)
    metrics_pykwavers = compute_trace_metrics(analytical, py_amp)
    metrics_between = compute_trace_metrics(kw_amp, py_amp)
    source_metrics = compute_trace_metrics(kw_source_weights, py_source_weights)

    return {
        "kwave": kwave,
        "pykwavers": pykwavers,
        "analytical": {
            "x_vec": x_vec,
            "amp_on_axis": analytical,
        },
        "source_metrics": source_metrics,
        "summary": metrics_between,
        "reference_metrics": {
            "kwave": metrics_kwave,
            "pykwavers": metrics_pykwavers,
        },
    }


_R_TARGET = 0.999
_RMS_MIN = 0.99
_RMS_MAX = 1.01


def main() -> int:
    """Execute the comparison, save diagnostics, and print metrics."""
    parser = argparse.ArgumentParser(description="at_circular_piston_3D parity compare.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    global REFRESH_CACHE
    if args.no_cache:
        REFRESH_CACHE = True

    result = run_comparison()
    summary = result["summary"]
    reference_metrics = result["reference_metrics"]
    x_vec = np.asarray(result["analytical"]["x_vec"], dtype=np.float64)
    analytical = np.asarray(result["analytical"]["amp_on_axis"], dtype=np.float64)

    r = float(summary["pearson_r"])
    rms = float(summary["rms_ratio"])
    overall_status = "PASS" if r >= _R_TARGET and _RMS_MIN <= rms <= _RMS_MAX else "FAIL"

    report_lines = build_report_lines(
        result["kwave"],
        result["pykwavers"],
        analytical,
        result["source_metrics"],
        reference_metrics["kwave"],
        reference_metrics["pykwavers"],
        summary,
    )
    report_lines.append(f"parity_status: {overall_status}")
    plot_comparison(result["kwave"], result["pykwavers"], x_vec, analytical)
    save_text_report(METRICS_PATH, "at_circular_piston_3D parity metrics", report_lines)

    print("=" * 80)
    print("at_circular_piston_3D: k-wave-python vs pykwavers")
    print("=" * 80)
    print(f"source weights Pearson r:          {result['source_metrics']['pearson_r']:.6f}")
    print(f"k-wave vs pykwavers Pearson r:     {r:.6f}  (target >= {_R_TARGET})")
    print(f"k-wave vs pykwavers RMS ratio:     {rms:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])")
    print(f"k-Wave vs analytical Pearson r:    {reference_metrics['kwave']['pearson_r']:.6f}")
    print(f"pykwavers vs analytical Pearson r: {reference_metrics['pykwavers']['pearson_r']:.6f}")
    print(f"Status:                            {overall_status}")
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {METRICS_PATH}")

    return 0 if overall_status == "PASS" or args.allow_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
