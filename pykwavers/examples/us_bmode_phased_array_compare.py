#!/usr/bin/env python3
"""
Parity comparison for the k-wave-python `us_bmode_phased_array` example.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    normalize_sensor_matrix,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.reconstruction.beamform import envelope_detection, scan_conversion
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.dotdictionary import dotdict
from kwave.utils.filters import gaussian_filter
from kwave.utils.matlab import matlab_find
from kwave.utils.mapgen import make_ball
from kwave.utils.signals import get_win, tone_burst
from pykwavers.parity_targets import evaluate_parity


PML_SIZE = Vector([15, 10, 10])
GRID_SIZE_POINTS = Vector([256, 256, 128]) - 2 * PML_SIZE
GRID_SIZE_METERS = 50e-3
GRID_SPACING_METERS = GRID_SIZE_METERS / Vector(
    [GRID_SIZE_POINTS.x, GRID_SIZE_POINTS.x, GRID_SIZE_POINTS.x]
)

C0 = 1540.0
RHO0 = 1000.0
ALPHA_COEFF = 0.75
ALPHA_POWER = 1.5
BON_A = 6.0
SOURCE_STRENGTH = 1e6
TONE_BURST_FREQ = 1e6
TONE_BURST_CYCLES = 4
STEERING_ANGLES_FULL = np.arange(-32, 33, 2)
STEERING_ANGLES_QUICK = np.arange(-32, 33, 8)
COMPRESSION_RATIO = 3

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "us_bmode_phased_array_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "us_bmode_phased_array_metrics.txt"


def build_reference_objects(seed: int):
    """Build deterministic phantom, grid, medium, and NotATransducer."""
    rng = np.random.default_rng(seed)

    kgrid = kWaveGrid(GRID_SIZE_POINTS, GRID_SPACING_METERS)
    t_end = (GRID_SIZE_POINTS.x * GRID_SPACING_METERS.x) * 2.2 / C0
    kgrid.makeTime(C0, t_end=t_end)

    medium = kWaveMedium(sound_speed=None, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER, BonA=BON_A)

    input_signal = tone_burst(1 / kgrid.dt, TONE_BURST_FREQ, TONE_BURST_CYCLES)
    input_signal = (SOURCE_STRENGTH / (C0 * RHO0)) * input_signal

    tr = dotdict()
    tr.number_elements = 64
    tr.element_width = 1
    tr.element_length = 40
    tr.element_spacing = 0
    tr.radius = float("inf")
    transducer_width = tr.number_elements * tr.element_width + (tr.number_elements - 1) * tr.element_spacing
    tr.position = np.round(
        [1, GRID_SIZE_POINTS.y / 2 - transducer_width / 2, GRID_SIZE_POINTS.z / 2 - tr.element_length / 2]
    )
    transducer = kWaveTransducerSimple(kgrid, **tr)

    nt = dotdict()
    nt.sound_speed = C0
    nt.focus_distance = 30e-3
    nt.elevation_focus_distance = 30e-3
    nt.steering_angle = 0
    nt.steering_angle_max = 32
    nt.transmit_apodization = "Rectangular"
    nt.receive_apodization = "Rectangular"
    nt.active_elements = np.ones((transducer.number_elements, 1))
    nt.input_signal = input_signal
    not_transducer = NotATransducer(transducer, kgrid, **nt)

    background_map = 1.0 + 0.008 * rng.standard_normal((kgrid.Nx, kgrid.Ny, kgrid.Nz))
    sound_speed_map = C0 * background_map
    density_map = RHO0 * background_map

    scattering_map = rng.standard_normal((kgrid.Nx, kgrid.Ny, kgrid.Nz))
    scattering_c0 = np.clip(C0 + 25 + 75 * scattering_map, 1400, 1600)
    scattering_rho0 = scattering_c0 / 1.5

    radius = 8e-3
    x_pos = 32e-3
    y_pos = kgrid.dy * kgrid.Ny / 2
    z_pos = kgrid.dz * kgrid.Nz / 2
    ball_center = np.round(Vector([x_pos, y_pos, z_pos]) / kgrid.dx)
    scattering_region = make_ball(GRID_SIZE_POINTS, ball_center, round(radius / kgrid.dx)).nonzero()
    sound_speed_map[scattering_region] = scattering_c0[scattering_region]
    density_map[scattering_region] = scattering_rho0[scattering_region]

    medium.sound_speed = sound_speed_map
    medium.density = density_map
    return kgrid, medium, transducer, not_transducer, np.asarray(input_signal).ravel()


def matlab_ordered_active_voxel_coords(not_transducer):
    """Return active transducer voxel coordinates in the same order used by k-wave-python."""
    active_mask = np.asarray(not_transducer.active_elements_mask)
    linear_indices = matlab_find(active_mask).astype(int).ravel() - 1
    coords = np.column_stack(np.unravel_index(linear_indices, active_mask.shape, order="F"))
    return coords.astype(int)


def c_order_active_voxel_coords(not_transducer):
    """Return active transducer voxel coordinates in ndarray C-order."""
    active_mask = np.asarray(not_transducer.active_elements_mask)
    return np.argwhere(active_mask != 0).astype(int)


def build_phased_source_mask_and_signals(kgrid, transducer, not_transducer, input_signal):
    """Build full-grid source mask and per-voxel velocity signals for pykwavers.

    k-wave-python's transducer internals use MATLAB/Fortran ordering, but
    pykwavers velocity-mask sources are expanded in ndarray C-order. The
    delay/apodization semantics come from the k-wave transducer object, while
    the returned signal rows are reordered into C-order so each voxel gets the
    correct delayed drive waveform inside pykwavers.
    """
    fnx = int(GRID_SIZE_POINTS.x) + 2 * int(PML_SIZE.x)
    fny = int(GRID_SIZE_POINTS.y) + 2 * int(PML_SIZE.y)
    fnz = int(GRID_SIZE_POINTS.z) + 2 * int(PML_SIZE.z)
    nt = int(kgrid.Nt)

    active_coords = c_order_active_voxel_coords(not_transducer)
    mask = build_phased_source_mask(active_coords)
    ux_signals = build_phased_velocity_signals(kgrid, not_transducer, input_signal, active_coords)
    return mask, ux_signals


def build_phased_source_mask(active_coords):
    """Build the fixed full-grid active-voxel mask for phased-array runs."""
    fnx = int(GRID_SIZE_POINTS.x) + 2 * int(PML_SIZE.x)
    fny = int(GRID_SIZE_POINTS.y) + 2 * int(PML_SIZE.y)
    fnz = int(GRID_SIZE_POINTS.z) + 2 * int(PML_SIZE.z)
    px, py, pz = int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)

    mask = np.zeros((fnx, fny, fnz), dtype=np.float64)
    for x_idx, y_idx, z_idx in active_coords:
        mask[x_idx + px, y_idx + py, z_idx + pz] = 1.0
    return mask


def build_phased_velocity_signals(kgrid, not_transducer, input_signal, active_coords=None):
    """Build per-voxel phased-array drive signals without allocating a full-grid mask."""
    if active_coords is None:
        active_coords = c_order_active_voxel_coords(not_transducer)

    delay_mask = np.asarray(not_transducer.delay_mask(), dtype=int)
    transmit_apod_mask = np.asarray(not_transducer.transmit_apodization_mask, dtype=np.float64)
    transducer_signal = np.asarray(not_transducer.input_signal, dtype=np.float64).reshape(-1)
    transducer_scale = 2.0 * float(not_transducer.sound_speed) * float(kgrid.dt) / float(kgrid.dx)

    nt = int(kgrid.Nt)
    ux_signals = np.zeros((active_coords.shape[0], nt), dtype=np.float64)

    for point, (x_idx, y_idx, z_idx) in enumerate(active_coords):
        delay = int(delay_mask[x_idx, y_idx, z_idx])
        weight = float(transmit_apod_mask[x_idx, y_idx, z_idx])
        start = delay
        end = min(delay + nt, transducer_signal.size)
        if end > start:
            ux_signals[point, : end - start] = transducer_signal[start:end] * weight * transducer_scale
    return ux_signals


def reorder_sensor_data_to_kwave_transducer_order(sensor_data_raw, not_transducer):
    """Reorder pykwavers sensor rows from C-order mask enumeration to k-wave-python MATLAB order."""
    raw = np.asarray(sensor_data_raw)
    coords_matlab = matlab_ordered_active_voxel_coords(not_transducer)
    coords_c = c_order_active_voxel_coords(not_transducer)
    c_index = {tuple(coord.tolist()): idx for idx, coord in enumerate(coords_c)}
    order = [c_index[tuple(coord.tolist())] for coord in coords_matlab]
    return raw[order, :]


def build_full_medium_arrays(sound_speed_map: np.ndarray, density_map: np.ndarray):
    """Embed the active-domain phantom into the full pykwavers grid including PML."""
    fnx = int(GRID_SIZE_POINTS.x) + 2 * int(PML_SIZE.x)
    fny = int(GRID_SIZE_POINTS.y) + 2 * int(PML_SIZE.y)
    fnz = int(GRID_SIZE_POINTS.z) + 2 * int(PML_SIZE.z)
    ss_full = np.full((fnx, fny, fnz), C0, dtype=np.float64)
    rho_full = np.full((fnx, fny, fnz), RHO0, dtype=np.float64)
    px, py, pz = int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)
    anx, any_, anz = int(GRID_SIZE_POINTS.x), int(GRID_SIZE_POINTS.y), int(GRID_SIZE_POINTS.z)
    ss_full[px : px + anx, py : py + any_, pz : pz + anz] = sound_speed_map.astype(np.float64)
    rho_full[px : px + anx, py : py + any_, pz : pz + anz] = density_map.astype(np.float64)
    return ss_full, rho_full


def run_kwave_phased_array(medium, kgrid, not_transducer, steering_angles, use_gpu: bool, cache_tag: str):
    """Run the k-wave-python reference phased-array scan-line loop."""
    cache_path = DEFAULT_OUTPUT_DIR / f"us_bmode_phased_array_kwave_{cache_tag}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["scan_lines"], float(cached["runtime_s"])

    scan_lines = np.zeros((len(steering_angles), kgrid.Nt))
    start = time.perf_counter()
    for angle_index, angle in enumerate(steering_angles):
        print(f"  [k-Wave {'GPU' if use_gpu else 'CPU'}] angle {angle_index + 1}/{len(steering_angles)} = {angle} deg")
        not_transducer.steering_angle = float(angle)
        sensor_data = kspaceFirstOrder3D(
            medium=deepcopy(medium),
            kgrid=kgrid,
            source=not_transducer,
            sensor=not_transducer,
            simulation_options=SimulationOptions(
                pml_inside=False,
                pml_size=PML_SIZE,
                data_cast="single",
                data_recast=True,
                save_to_disk=True,
                save_to_disk_exit=False,
            ),
            execution_options=SimulationExecutionOptions(is_gpu_simulation=use_gpu),
        )
        scan_lines[angle_index, :] = not_transducer.scan_line(not_transducer.combine_sensor_data(sensor_data["p"].T))
    runtime_s = time.perf_counter() - start
    np.savez(cache_path, scan_lines=scan_lines, runtime_s=runtime_s)
    return scan_lines, runtime_s


def run_pykwavers_phased_array(
    sound_speed_map,
    density_map,
    kgrid,
    transducer,
    not_transducer,
    input_signal,
    steering_angles,
    use_gpu: bool,
    cache_tag: str,
):
    """Run the pykwavers phased-array scan-line loop with identical beamforming/post-beamforming."""
    cache_path = DEFAULT_OUTPUT_DIR / f"us_bmode_phased_array_pykwavers_{cache_tag}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        profile = None
        if "profile_total_ns" in cached.files:
            profile = {
                "total_ns": cached["profile_total_ns"],
                "solver_run_ns": cached["profile_solver_run_ns"],
                "materialize_ns": cached["profile_materialize_ns"],
                "medium_upload_ns": cached["profile_medium_upload_ns"],
            }
        return cached["scan_lines"], float(cached["runtime_s"]), profile

    fnx = int(GRID_SIZE_POINTS.x) + 2 * int(PML_SIZE.x)
    fny = int(GRID_SIZE_POINTS.y) + 2 * int(PML_SIZE.y)
    fnz = int(GRID_SIZE_POINTS.z) + 2 * int(PML_SIZE.z)
    grid = pkw.Grid(fnx, fny, fnz, GRID_SPACING_METERS.x, GRID_SPACING_METERS.y, GRID_SPACING_METERS.z)
    ss_full, rho_full = build_full_medium_arrays(sound_speed_map, density_map)
    medium = pkw.Medium(
        sound_speed=ss_full,
        density=rho_full,
        absorption=np.full((fnx, fny, fnz), ALPHA_COEFF, dtype=np.float64),
        nonlinearity=np.full((fnx, fny, fnz), BON_A, dtype=np.float64),
    )

    scan_lines = np.zeros((len(steering_angles), kgrid.Nt))
    profile_totals_ns = np.zeros(len(steering_angles), dtype=np.float64)
    profile_solver_ns = np.zeros(len(steering_angles), dtype=np.float64)
    profile_materialize_ns = np.zeros(len(steering_angles), dtype=np.float64)
    profile_upload_ns = np.zeros(len(steering_angles), dtype=np.float64)
    start = time.perf_counter()
    gpu_session = None
    expected_sensor_points = int(not_transducer.number_active_elements * transducer.element_width * transducer.element_length)
    absorption_full = np.full((fnx, fny, fnz), ALPHA_COEFF, dtype=np.float64)
    nonlinearity_full = np.full((fnx, fny, fnz), BON_A, dtype=np.float64)
    if use_gpu:
        active_coords = c_order_active_voxel_coords(not_transducer)
        gpu_session = pkw.GpuPstdSession(
            grid,
            ss_full,
            rho_full,
            dt=kgrid.dt,
            time_steps=kgrid.Nt,
            absorption=absorption_full,
            nonlinearity=nonlinearity_full,
            pml_size_xyz=(int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)),
            alpha_power=ALPHA_POWER,
        )
        base_mask = build_phased_source_mask(active_coords)
        gpu_session.set_source_sensor_mask(base_mask.astype(np.float64))
    for angle_index, angle in enumerate(steering_angles):
        print(f"  [pykwavers {'GPU' if use_gpu else 'CPU'}] angle {angle_index + 1}/{len(steering_angles)} = {angle} deg")
        not_transducer.steering_angle = float(angle)
        if use_gpu:
            ux_signals = build_phased_velocity_signals(
                kgrid,
                not_transducer,
                input_signal,
                active_coords,
            )
            gpu_session.set_velocity_signals(ux_signals)
            sensor_data_raw = normalize_sensor_matrix(
                np.asarray(gpu_session.run_scan_line_cached()),
                expected_sensors=expected_sensor_points,
            )
            profile = dict(gpu_session.last_run_profile)
            profile_totals_ns[angle_index] = float(profile["total_ns"])
            profile_solver_ns[angle_index] = float(profile["solver_run_ns"])
            profile_materialize_ns[angle_index] = float(profile["materialize_ns"])
            profile_upload_ns[angle_index] = float(profile["medium_upload_ns"])
            # pykwavers SensorRecorder returns data in MATLAB Fortran order (y-fast for
            # fixed-x transducer), which is the same order combine_sensor_data expects.
            # No reordering needed.
        else:
            mask, ux_signals = build_phased_source_mask_and_signals(kgrid, transducer, not_transducer, input_signal)
            source = pkw.Source.from_velocity_mask_2d(mask, ux=ux_signals, mode="additive")
            sensor = pkw.Sensor.from_mask(mask.astype(bool))
            sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
            sim.set_pml_size_xyz(int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z))
            sim.set_nonlinear(True)
            sim.set_alpha_coeff(ALPHA_COEFF)
            sim.set_alpha_power(ALPHA_POWER)
            result = sim.run(kgrid.Nt, dt=kgrid.dt)
            sensor_data_raw = normalize_sensor_matrix(
                np.asarray(result.sensor_data),
                expected_sensors=expected_sensor_points,
            )
            # pykwavers SensorRecorder returns data in MATLAB Fortran order (y-fast for
            # fixed-x transducer), which is the same order combine_sensor_data expects.
            # No reordering needed.
        scan_lines[angle_index, :] = not_transducer.scan_line(not_transducer.combine_sensor_data(sensor_data_raw))
    runtime_s = time.perf_counter() - start
    profile = None
    if use_gpu:
        profile = {
            "total_ns": profile_totals_ns,
            "solver_run_ns": profile_solver_ns,
            "materialize_ns": profile_materialize_ns,
            "medium_upload_ns": profile_upload_ns,
        }
        np.savez(
            cache_path,
            scan_lines=scan_lines,
            runtime_s=runtime_s,
            profile_total_ns=profile_totals_ns,
            profile_solver_run_ns=profile_solver_ns,
            profile_materialize_ns=profile_materialize_ns,
            profile_medium_upload_ns=profile_upload_ns,
        )
    else:
        np.savez(cache_path, scan_lines=scan_lines, runtime_s=runtime_s)
    return scan_lines, runtime_s, profile


def summarize_gpu_profile(profile: dict[str, np.ndarray] | None) -> list[str]:
    """Summarize aggregated per-angle GPU timing data in milliseconds."""
    if not profile:
        return []

    def stats_line(label: str, values_ns: np.ndarray) -> str:
        values_ms = np.asarray(values_ns, dtype=float) / 1e6
        return (
            f"  {label}: mean_ms = {np.mean(values_ms):.3f}, "
            f"max_ms = {np.max(values_ms):.3f}, min_ms = {np.min(values_ms):.3f}"
        )

    return [
        "gpu_profile:",
        stats_line("total", profile["total_ns"]),
        stats_line("solver_run", profile["solver_run_ns"]),
        stats_line("materialize", profile["materialize_ns"]),
        stats_line("medium_upload", profile["medium_upload_ns"]),
    ]


def post_process(scan_lines, kgrid, medium, not_transducer, steering_angles):
    """Run the same phased-array post-processing used by the k-wave-python example."""
    t0_offset = int(round(len(not_transducer.input_signal.squeeze()) / 2) + (not_transducer.appended_zeros - not_transducer.beamforming_delays_offset))
    scan_lines = scan_lines[:, t0_offset:]
    nt = scan_lines.shape[1]

    tukey_win, _ = get_win(nt * 2, "Tukey", False, 0.05)
    scan_line_win = np.concatenate(
        (np.zeros([1, t0_offset * 2]), tukey_win.T[:, : int(len(tukey_win) / 2) - t0_offset * 2]),
        axis=1,
    )
    scan_lines = scan_lines * scan_line_win

    r = C0 * np.arange(1, nt + 1) * kgrid.dt / 2
    tgc_alpha_db_cm = medium.alpha_coeff * (TONE_BURST_FREQ * 1e-6) ** medium.alpha_power
    tgc_alpha_np_m = db2neper(tgc_alpha_db_cm) * 100
    tgc = np.exp(tgc_alpha_np_m * 2 * r)
    scan_lines *= tgc

    scan_lines_fund = gaussian_filter(scan_lines, 1 / kgrid.dt, TONE_BURST_FREQ, 100)
    scan_lines_harm = gaussian_filter(scan_lines, 1 / kgrid.dt, 2 * TONE_BURST_FREQ, 30)
    scan_lines_fund = envelope_detection(scan_lines_fund)
    scan_lines_harm = envelope_detection(scan_lines_harm)
    scan_lines_fund = log_compression(scan_lines_fund, COMPRESSION_RATIO, True)
    scan_lines_harm = log_compression(scan_lines_harm, COMPRESSION_RATIO, True)

    image_size = [kgrid.Nx * kgrid.dx, kgrid.Ny * kgrid.dy]
    image_res = [256, 256]
    b_mode_fund = scan_conversion(scan_lines_fund, steering_angles, image_size, C0, kgrid.dt, image_res)
    b_mode_harm = scan_conversion(scan_lines_harm, steering_angles, image_size, C0, kgrid.dt, image_res)
    return scan_lines, scan_lines_fund, scan_lines_harm, b_mode_fund, b_mode_harm


def plot_comparison(kwave: dict, pykwavers: dict, steering_angles) -> None:
    """Save 2x3 phased-array comparison figure."""
    image_size = [GRID_SIZE_POINTS.x * GRID_SPACING_METERS.x, GRID_SIZE_POINTS.y * GRID_SPACING_METERS.y]
    x_axis = [0, image_size[0] * 1e3]
    y_axis = [0, image_size[1] * 1e3]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rows = [
        ("k-wave-python", kwave["raw"], kwave["fund"], kwave["bmode_fund"]),
        ("pykwavers", pykwavers["raw"], pykwavers["fund"], pykwavers["bmode_fund"]),
    ]

    for row_idx, (label, raw, fund, bmode) in enumerate(rows):
        axes[row_idx, 0].imshow(raw.T, aspect="auto", extent=[steering_angles[-1], steering_angles[0], y_axis[1], y_axis[0]], interpolation="none", cmap="gray")
        axes[row_idx, 0].set_title(f"Raw Scan Lines ({label})")
        axes[row_idx, 0].set_xlabel("Steering angle [deg]")
        axes[row_idx, 0].set_ylabel("Depth [mm]")

        axes[row_idx, 1].imshow(fund.T, aspect="auto", extent=[steering_angles[-1], steering_angles[0], y_axis[1], y_axis[0]], interpolation="none", cmap="bone")
        axes[row_idx, 1].set_title(f"Processed Fundamental ({label})")
        axes[row_idx, 1].set_xlabel("Steering angle [deg]")
        axes[row_idx, 1].set_ylabel("Depth [mm]")

        axes[row_idx, 2].imshow(bmode, cmap="bone", aspect="auto", extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]], interpolation="none")
        axes[row_idx, 2].set_title(f"B-Mode Image ({label})")
        axes[row_idx, 2].set_xlabel("Horizontal Position [mm]")
        axes[row_idx, 2].set_ylabel("Depth [mm]")

    fig.suptitle("us_bmode_phased_array: k-wave-python vs pykwavers", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_report_lines(kwave: dict, pykwavers: dict, steering_angles) -> list[str]:
    """Build plain-text metrics report."""
    metrics_fund = compute_image_metrics(kwave["bmode_fund"], pykwavers["bmode_fund"])
    metrics_harm = compute_image_metrics(kwave["bmode_harm"], pykwavers["bmode_harm"])
    eval_fund = evaluate_parity(metrics_fund, "fundamental", len(steering_angles), len(STEERING_ANGLES_FULL))
    eval_harm = evaluate_parity(metrics_harm, "harmonic", len(steering_angles), len(STEERING_ANGLES_FULL))
    overall = "PASS" if eval_fund["status"] == "PASS" and eval_harm["status"] == "PASS" else "FAIL"

    lines = [
        f"parity_status: {overall}",
        "example: us_bmode_phased_array",
        f"steering_angles: {len(steering_angles)}/{len(STEERING_ANGLES_FULL)}",
        f"kwave_runtime_s: {kwave['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pykwavers['runtime_s']:.3f}",
        "",
        "fundamental:",
        f"  pearson_r = {metrics_fund['pearson_r']:.6f}",
        f"  rms_ratio = {metrics_fund['rms_ratio']:.6f}",
        f"  psnr_db   = {metrics_fund['psnr_db']:.3f}",
        f"  tier      = {eval_fund['tier']}",
        f"  status    = {eval_fund['status']}",
        "",
        "harmonic:",
        f"  pearson_r = {metrics_harm['pearson_r']:.6f}",
        f"  rms_ratio = {metrics_harm['rms_ratio']:.6f}",
        f"  psnr_db   = {metrics_harm['psnr_db']:.3f}",
        f"  tier      = {eval_harm['tier']}",
        f"  status    = {eval_harm['status']}",
    ]
    lines.extend(summarize_gpu_profile(pykwavers.get("gpu_profile")))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pykwavers with k-wave-python for us_bmode_phased_array.")
    parser.add_argument("--quick", action="store_true", help="Use a reduced steering-angle set for faster diagnostics.")
    parser.add_argument("--seed", type=int, default=20260401, help="Deterministic phantom RNG seed.")
    parser.add_argument("--kwave-gpu", action="store_true", help="Run k-wave-python with the CUDA binary.")
    parser.add_argument("--pykwavers-gpu", action="store_true", help="Run pykwavers with GpuPstdSession.")
    args = parser.parse_args()

    steering_angles = STEERING_ANGLES_QUICK if args.quick else STEERING_ANGLES_FULL
    cache_tag = (
        f"{'quick' if args.quick else 'full'}_seed{args.seed}_kw{'gpu' if args.kwave_gpu else 'cpu'}_pkw{'gpu' if args.pykwavers_gpu else 'cpu'}"
    )
    kgrid, medium, transducer, not_transducer, input_signal = build_reference_objects(args.seed)

    kwave_scan_lines, kwave_runtime = run_kwave_phased_array(
        medium, kgrid, not_transducer, steering_angles, args.kwave_gpu, cache_tag
    )
    kw_raw, kw_fund, kw_harm, kw_bmode_fund, kw_bmode_harm = post_process(kwave_scan_lines.copy(), kgrid, medium, not_transducer, steering_angles)

    pkw_scan_lines, pkw_runtime, pkw_gpu_profile = run_pykwavers_phased_array(
        medium.sound_speed,
        medium.density,
        kgrid,
        transducer,
        not_transducer,
        input_signal,
        steering_angles,
        args.pykwavers_gpu,
        cache_tag,
    )
    pkw_raw, pkw_fund, pkw_harm, pkw_bmode_fund, pkw_bmode_harm = post_process(pkw_scan_lines.copy(), kgrid, medium, not_transducer, steering_angles)

    kwave_bundle = {
        "runtime_s": kwave_runtime,
        "raw": kw_raw,
        "fund": kw_fund,
        "harm": kw_harm,
        "bmode_fund": kw_bmode_fund,
        "bmode_harm": kw_bmode_harm,
    }
    pykwavers_bundle = {
        "runtime_s": pkw_runtime,
        "raw": pkw_raw,
        "fund": pkw_fund,
        "harm": pkw_harm,
        "bmode_fund": pkw_bmode_fund,
        "bmode_harm": pkw_bmode_harm,
        "gpu_profile": pkw_gpu_profile,
    }

    plot_comparison(kwave_bundle, pykwavers_bundle, steering_angles)
    save_text_report(METRICS_PATH, "us_bmode_phased_array parity metrics", build_report_lines(kwave_bundle, pykwavers_bundle, steering_angles))
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
