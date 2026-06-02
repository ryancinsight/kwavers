#!/usr/bin/env python3
"""
Parity comparison for the k-wave-python `us_defining_transducer` example.
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_trace_metrics,
    normalize_sensor_matrix,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kWaveSimulation import SimulationOptions
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.dotdictionary import dotdict
from kwave.utils.filters import spect
from kwave.utils.signals import tone_burst
PML_X_SIZE = 20
PML_Y_SIZE = 10
PML_Z_SIZE = 10
NX = 128 - 2 * PML_X_SIZE
NY = 128 - 2 * PML_Y_SIZE
NZ = 64 - 2 * PML_Z_SIZE
TOTAL_NX = NX + 2 * PML_X_SIZE
TOTAL_NY = NY + 2 * PML_Y_SIZE
TOTAL_NZ = NZ + 2 * PML_Z_SIZE

X_LEN = 40e-3
DX = X_LEN / NX
DY = DX
DZ = DX
SOUND_SPEED = 1540.0
DENSITY = 1000.0
T_END = 40e-6
SOURCE_STRENGTH = 1e6
TONE_BURST_FREQ = 0.5e6
TONE_BURST_CYCLES = 5

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "us_defining_transducer_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "us_defining_transducer_metrics.txt"
CACHE_VERSION = 2
TRACE_THRESHOLDS = {
    "pearson_r": 0.90,
    "rms_ratio_min": 0.80,
    "rms_ratio_max": 1.25,
}

def build_kwave_configuration():
    """Build the canonical k-wave-python transducer configuration."""
    kgrid = kWaveGrid([NX, NY, NZ], [DX, DY, DZ])
    medium = kWaveMedium(
        sound_speed=SOUND_SPEED,
        density=DENSITY,
        alpha_coeff=0.75,
        alpha_power=1.5,
        BonA=6.0,
    )
    kgrid.makeTime(medium.sound_speed, t_end=T_END)

    input_signal = tone_burst(1 / kgrid.dt, TONE_BURST_FREQ, TONE_BURST_CYCLES)
    input_signal = (SOURCE_STRENGTH / (medium.sound_speed * medium.density)) * input_signal

    transducer = dotdict()
    transducer.number_elements = 72
    transducer.element_width = 1
    transducer.element_length = 12
    transducer.element_spacing = 0
    transducer.radius = np.inf
    transducer.position = np.round([1, NY / 2 - NX // 2, NZ / 2 - transducer.element_length / 2])
    transducer = kWaveTransducerSimple(kgrid, **transducer)

    not_transducer = dotdict()
    not_transducer.sound_speed = medium.sound_speed
    not_transducer.focus_distance = 20e-3
    not_transducer.elevation_focus_distance = 19e-3
    not_transducer.steering_angle = 0
    not_transducer.transmit_apodization = "Rectangular"
    not_transducer.receive_apodization = "Rectangular"
    not_transducer.active_elements = np.zeros((transducer.number_elements, 1))
    not_transducer.active_elements[21:52] = 1
    not_transducer.input_signal = input_signal
    not_transducer = NotATransducer(transducer, kgrid, **not_transducer)

    sensor_mask = np.zeros((NX, NY, NZ))
    sensor_mask[NX // 4, NY // 2, NZ // 2] = 1
    sensor_mask[NX // 2, NY // 2, NZ // 2] = 1
    sensor_mask[3 * NX // 4, NY // 2, NZ // 2] = 1
    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]

    return kgrid, medium, not_transducer, sensor, np.asarray(input_signal)
def run_kwave_reference(use_gpu: bool) -> dict:
    """Run the k-wave-python reference simulation."""
    kwave_cache_path = DEFAULT_OUTPUT_DIR / (
        "us_defining_transducer_kwave_cache_gpu.npz" if use_gpu else "us_defining_transducer_kwave_cache_cpu.npz"
    )
    if kwave_cache_path.exists():
        cached = np.load(kwave_cache_path)
        cache_version = int(np.asarray(cached["cache_version"]).reshape(())) if "cache_version" in cached.files else 0
        if cache_version == CACHE_VERSION:
            return {
                "pressure": cached["pressure"],
                "time": cached["time"],
                "dt": float(cached["dt"]),
                "runtime_s": float(cached["runtime_s"]),
                "input_signal": cached["input_signal"],
                "active_mask": cached["active_mask"],
                "time_steps": int(cached["time_steps"]),
            }

    kgrid, medium, not_transducer, sensor, input_signal = build_kwave_configuration()
    start = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium,
        kgrid=kgrid,
        source=not_transducer,
        sensor=sensor,
        simulation_options=SimulationOptions(
            pml_inside=False,
            pml_size=Vector([PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE]),
            data_cast="single",
            save_to_disk=True,
        ),
        execution_options=SimulationExecutionOptions(is_gpu_simulation=use_gpu),
    )
    elapsed = time.perf_counter() - start
    pressure = normalize_sensor_matrix(sensor_data["p"])
    result = {
        "pressure": pressure,
        "time": np.asarray(kgrid.t_array).ravel()[: pressure.shape[1]],
        "dt": float(kgrid.dt),
        "runtime_s": elapsed,
        "input_signal": input_signal.ravel(),
        "active_mask": np.asarray(not_transducer.all_elements_mask),
        "time_steps": int(pressure.shape[1]),
    }
    np.savez(
        kwave_cache_path,
        cache_version=CACHE_VERSION,
        pressure=result["pressure"],
        time=result["time"],
        dt=result["dt"],
        runtime_s=result["runtime_s"],
        input_signal=result["input_signal"],
        active_mask=result["active_mask"],
        time_steps=result["time_steps"],
    )
    return result
def build_pykwavers_inputs(dt: float, input_signal: np.ndarray, n_steps: int):
    """Construct pykwavers source/sensor inputs matching the k-wave transducer example."""
    grid = pkw.Grid(TOTAL_NX, TOTAL_NY, TOTAL_NZ, DX, DY, DZ)
    medium = pkw.Medium.homogeneous(sound_speed=SOUND_SPEED, density=DENSITY)

    active_mask = np.zeros(72, dtype=bool)
    active_mask[21:52] = True
    active_indices = np.where(active_mask)[0]

    focus_distance = 20e-3
    elevation_focus_distance = 19e-3
    element_length = 12
    element_pitch = DY
    element_index_az = np.arange(-(len(active_indices) - 1) / 2, (len(active_indices) + 1) / 2)
    az_delays_s = (focus_distance / SOUND_SPEED) * (
        1 - np.sqrt(1 + (element_index_az * element_pitch / focus_distance) ** 2)
    )
    az_delays = -np.round(az_delays_s / dt).astype(int)

    elev_index = np.arange(-(element_length - 1) / 2, (element_length + 1) / 2)
    elev_delays_s = (
        elevation_focus_distance
        - np.sqrt((elev_index * DZ) ** 2 + elevation_focus_distance**2)
    ) / SOUND_SPEED
    elev_delays = -np.round(elev_delays_s / dt).astype(int)

    total_delays = np.zeros(len(active_indices) * element_length, dtype=int)
    for el_idx in range(len(active_indices)):
        for ev_idx in range(element_length):
            total_delays[el_idx * element_length + ev_idx] = az_delays[el_idx] + elev_delays[ev_idx]
    total_delays -= int(total_delays.min())

    source_mask = np.zeros((TOTAL_NX, TOTAL_NY, TOTAL_NZ), dtype=np.float64)
    sensor_mask = np.zeros((TOTAL_NX, TOTAL_NY, TOTAL_NZ), dtype=bool)
    sensor_positions = [
        (NX // 4 + PML_X_SIZE, NY // 2 + PML_Y_SIZE, NZ // 2 + PML_Z_SIZE),
        (NX // 2 + PML_X_SIZE, NY // 2 + PML_Y_SIZE, NZ // 2 + PML_Z_SIZE),
        (3 * NX // 4 + PML_X_SIZE, NY // 2 + PML_Y_SIZE, NZ // 2 + PML_Z_SIZE),
    ]
    for pos in sensor_positions:
        sensor_mask[pos] = True

    offset_x = PML_X_SIZE
    offset_y = int(NY / 2 - NX // 2) - 1 + PML_Y_SIZE
    offset_z = int(NZ / 2 - element_length // 2) - 1 + PML_Z_SIZE

    # k-Wave's NotATransducer internally prepends stored_appended_zeros = max_delay zeros
    # to the input_signal (input_signal property in ktransducer.py).  This shifts the
    # entire injection by max_delay steps so that every element has "room" to read ahead.
    # Injection rule: at step t, element with delay d injects padded_signal[t + d].
    # Center (delay=0): injects original_signal[0] at t = max_delay.
    # Outer (delay=8): injects original_signal[0] at t = max_delay - 8 = 1.
    # We replicate this exactly:
    max_delay = int(total_delays.max())
    padded_signal = np.concatenate([np.zeros(max_delay), input_signal.ravel()])

    ux = np.zeros((len(active_indices) * element_length, n_steps), dtype=np.float64)

    # kwavers applies 2*c0*dt/dx internally for additive velocity sources
    # (commit caabc640). Do NOT apply the factor here.

    point_index = 0
    for global_el in active_indices:
        for elev_idx in range(element_length):
            iy = offset_y + global_el
            iz = offset_z + elev_idx
            source_mask[offset_x, iy, iz] = 1.0
            delay = total_delays[point_index]
            # Read-ahead injection: at step t, inject padded_signal[t + delay].
            # Larger delay = reads further ahead = fires original signal EARLIER.
            n_inj = min(padded_signal.size - delay, n_steps)
            if n_inj > 0:
                ux[point_index, :n_inj] = padded_signal[delay : delay + n_inj]
            point_index += 1

    # Use additive mode to match k-Wave's NotATransducer source injection semantics.
    source = pkw.Source.from_velocity_mask_2d(source_mask, ux=ux, mode="additive")
    sensor = pkw.Sensor.from_mask(sensor_mask)
    return grid, medium, source, sensor
def run_pykwavers(dt: float, input_signal: np.ndarray, n_steps: int) -> dict:
    """Run the pykwavers counterpart simulation."""
    pykwavers_cache_path = DEFAULT_OUTPUT_DIR / "us_defining_transducer_pykwavers_cache.npz"
    if pykwavers_cache_path.exists():
        cached = np.load(pykwavers_cache_path)
        cache_version = int(np.asarray(cached["cache_version"]).reshape(())) if "cache_version" in cached.files else 0
        if cache_version == CACHE_VERSION and abs(float(cached["dt"]) - dt) < 1e-15:
            return {
                "pressure": cached["pressure"],
                "time": cached["time"],
                "runtime_s": float(cached["runtime_s"]),
                "time_steps": int(cached["time_steps"]),
            }

    grid, medium, source, sensor = build_pykwavers_inputs(dt, input_signal, n_steps)
    start = time.perf_counter()
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size_xyz(PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE)
    sim.set_pml_inside(True)
    result = sim.run(time_steps=n_steps, dt=dt)
    elapsed = time.perf_counter() - start
    pressure = normalize_sensor_matrix(np.asarray(result.sensor_data))
    time_axis = np.arange(pressure.shape[1], dtype=float) * dt
    output = {
        "pressure": pressure,
        "time": time_axis,
        "runtime_s": elapsed,
        "time_steps": int(n_steps),
    }
    np.savez(
        pykwavers_cache_path,
        cache_version=CACHE_VERSION,
        pressure=output["pressure"],
        time=output["time"],
        runtime_s=output["runtime_s"],
        dt=dt,
        time_steps=output["time_steps"],
    )
    return output
def trace_spectrum(trace: np.ndarray, dt: float):
    """Return single-sided spectrum for plotting."""
    f, amp, _phase = spect(np.asarray(trace).ravel(), 1.0 / dt)
    return np.asarray(f).ravel(), np.asarray(amp).ravel()
def plot_comparison(kwave: dict, pykwavers: dict, dt: float) -> None:
    """Save comparison figure for aperture geometry, traces, and spectra."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    aperture_plane = kwave["active_mask"][0, :, :]
    axes[0].imshow(aperture_plane.T, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Active Transducer Aperture")
    axes[0].set_xlabel("y [grid index]")
    axes[0].set_ylabel("z [grid index]")

    labels = ["Sensor 1", "Sensor 2", "Sensor 3"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for idx, color in enumerate(colors):
        axes[1].plot(kwave["time"] * 1e6, kwave["pressure"][idx], color=color, linewidth=1.5, label=f"{labels[idx]} k-Wave")
        axes[1].plot(pykwavers["time"] * 1e6, pykwavers["pressure"][idx], color=color, linewidth=1.2, linestyle="--", label=f"{labels[idx]} pykwavers")
    axes[1].set_title("Pressure Traces")
    axes[1].set_xlabel("Time [us]")
    axes[1].set_ylabel("Pressure [Pa]")
    axes[1].legend(fontsize=7, ncol=2)

    for idx, color in enumerate(colors):
        f_kw, amp_kw = trace_spectrum(kwave["pressure"][idx], dt)
        f_py, amp_py = trace_spectrum(pykwavers["pressure"][idx], dt)
        amp_kw = amp_kw / (np.max(amp_kw) + 1e-30)
        amp_py = amp_py / (np.max(amp_py) + 1e-30)
        axes[2].plot(f_kw * 1e-6, amp_kw, color=color, linewidth=1.5, label=f"{labels[idx]} k-Wave")
        axes[2].plot(f_py * 1e-6, amp_py, color=color, linewidth=1.2, linestyle="--", label=f"{labels[idx]} pykwavers")
    axes[2].axvline(TONE_BURST_FREQ * 1e-6, color="black", linestyle=":", linewidth=1.0)
    axes[2].set_xlim(0.0, SOUND_SPEED / (2 * DX) * 1e-6)
    axes[2].set_title("Normalized Spectra")
    axes[2].set_xlabel("Frequency [MHz]")
    axes[2].set_ylabel("Amplitude [a.u.]")

    fig.suptitle("us_defining_transducer: k-wave-python vs pykwavers", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
def build_report_lines(kwave: dict, pykwavers: dict) -> list[str]:
    """Build plain-text metrics report."""
    lines = [
        "example: us_defining_transducer",
        f"grid_total: {TOTAL_NX}x{TOTAL_NY}x{TOTAL_NZ}",
        f"dt_s: {kwave['dt']:.9e}",
        f"kwave_runtime_s: {kwave['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pykwavers['runtime_s']:.3f}",
        "",
        "thresholds:",
        f"  pearson_r >= {TRACE_THRESHOLDS['pearson_r']:.2f}",
        f"  {TRACE_THRESHOLDS['rms_ratio_min']:.2f} <= rms_ratio <= {TRACE_THRESHOLDS['rms_ratio_max']:.2f}",
        "",
    ]

    statuses = []
    for idx in range(min(kwave["pressure"].shape[0], pykwavers["pressure"].shape[0])):
        metrics = compute_trace_metrics(kwave["pressure"][idx], pykwavers["pressure"][idx])
        passed = (
            metrics["pearson_r"] >= TRACE_THRESHOLDS["pearson_r"]
            and TRACE_THRESHOLDS["rms_ratio_min"] <= metrics["rms_ratio"] <= TRACE_THRESHOLDS["rms_ratio_max"]
        )
        statuses.append(passed)
        lines.extend(
            [
                f"sensor_{idx + 1}: {'PASS' if passed else 'FAIL'}",
                f"  pearson_r    = {metrics['pearson_r']:.6f}",
                f"  rms_ratio    = {metrics['rms_ratio']:.6f}",
                f"  rmse         = {metrics['rmse']:.6e}",
                f"  max_abs_diff = {metrics['max_abs_diff']:.6e}",
                f"  peak_ratio   = {metrics['peak_ratio']:.6f}",
                "",
            ]
        )
    lines.insert(0, f"parity_status: {'PASS' if all(statuses) else 'FAIL'}")
    return lines
def run_comparison(use_gpu: bool = False) -> dict[str, dict[str, np.ndarray | float]]:
    """Run both reference simulations and return their cached-or-fresh outputs."""
    kwave = run_kwave_reference(use_gpu=use_gpu)
    pykwavers = run_pykwavers(kwave["dt"], kwave["input_signal"], int(kwave["time_steps"]))
    return {"kwave": kwave, "pykwavers": pykwavers}
def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pykwavers with k-wave-python for us_defining_transducer.")
    parser.add_argument("--gpu", action="store_true", help="Run k-wave-python with GPU execution if available.")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    comparison = run_comparison(use_gpu=args.gpu)
    kwave = comparison["kwave"]
    pykwavers = comparison["pykwavers"]
    plot_comparison(kwave, pykwavers, kwave["dt"])
    lines = build_report_lines(kwave, pykwavers)
    save_text_report(METRICS_PATH, "us_defining_transducer parity metrics", lines)
    for line in lines:
        print(line)
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {METRICS_PATH}")
    passed = any("parity_status: PASS" in str(l) for l in lines)
    return 0 if passed or args.allow_failure else 1

if __name__ == "__main__":
    raise SystemExit(main())
