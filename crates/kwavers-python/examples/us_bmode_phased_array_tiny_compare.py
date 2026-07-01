#!/usr/bin/env python3
"""
Tiny phased-array parity comparison: exercises the FULL stack
(NotATransducer delay/apodization, combine_sensor_data, scan_line)
on a small grid so k-wave-python finishes in minutes rather than ~90 min.

Grid: 64 x 128 x 32  (total, including PML)
PML:  [10, 10, 10]
Active: 44 x 108 x 12  grid points
Elements: 8, element_width=1, element_length=4
Sensor voxels: 8 * 1 * 4 = 32

Purpose: reproduces the full phased-array parity machinery (the code path
that showed rms_ratio=1.442 in the production 256x256x128 run) in a
fraction of the time, to isolate whether the bug is in:
  (a) apodization / delay aggregation in build_phased_velocity_signals
  (b) combine_sensor_data row ordering / elevation delay application
  (c) scale-dependent effects at large grid size

Outputs:
  output/us_bmode_phased_array_tiny_compare.png   — scan-line overlay
  output/us_bmode_phased_array_tiny_metrics.txt   — rms_ratio / pearson_r
"""

from __future__ import annotations

import argparse
import os
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
    compute_trace_metrics,
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
from kwave.reconstruction.beamform import envelope_detection
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.dotdictionary import dotdict
from kwave.utils.filters import gaussian_filter
from kwave.utils.matlab import matlab_find
from kwave.utils.signals import get_win, tone_burst

# ── Grid constants ────────────────────────────────────────────────────────────
PML_SIZE = Vector([10, 10, 10])
TOTAL_GRID = Vector([64, 128, 32])
ACTIVE_GRID = TOTAL_GRID - 2 * PML_SIZE
GRID_SIZE_METERS = 50e-3  # same dx as full example → dx ≈ 1.95e-4 m
GRID_SPACING_METERS = GRID_SIZE_METERS / Vector([256, 256, 256])  # match full-example dx

C0 = 1540.0
RHO0 = 1000.0
ALPHA_COEFF = 0.75
ALPHA_POWER = 1.5
BON_A = 6.0
SOURCE_STRENGTH = 1e6
TONE_BURST_FREQ = 1e6
TONE_BURST_CYCLES = 4

# Tiny transducer
N_ELEMENTS = 8
ELEMENT_WIDTH = 1    # voxels
ELEMENT_LENGTH = 4   # voxels
ELEMENT_SPACING = 0

STEERING_ANGLES = np.arange(-16, 17, 8)  # 5 angles: -16 -8 0 8 16

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PNG_PATH = OUTPUT_DIR / "us_bmode_phased_array_tiny_compare.png"
METRICS_PATH = OUTPUT_DIR / "us_bmode_phased_array_tiny_metrics.txt"
KWAVE_CACHE = OUTPUT_DIR / "us_bmode_phased_array_tiny_kwave_cache.npz"
PYKWAVERS_CACHE = OUTPUT_DIR / "us_bmode_phased_array_tiny_pykwavers_cache.npz"
REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1
PARITY_THRESHOLDS: dict[str, float] = {
    "mean_pearson_r": 0.98,
    "mean_rms_ratio_min": 0.90,
    "mean_rms_ratio_max": 1.10,
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_scan_line_cache(
    path: os.PathLike[str],
    steering_angles: np.ndarray,
    nt: int,
    seed: int,
) -> dict | None:
    """Load cached scan-line NPZ. Returns None if absent, stale, or REFRESH_CACHE."""
    if REFRESH_CACHE:
        return None
    cache_path = os.fspath(path)
    if not os.path.exists(cache_path):
        return None
    cached = np.load(cache_path, allow_pickle=False)
    if int(np.asarray(cached.get("cache_version", np.array(0))).reshape(())) != CACHE_VERSION:
        return None
    if int(np.asarray(cached["nt"]).reshape(())) != nt:
        return None
    if int(np.asarray(cached["seed"]).reshape(())) != seed:
        return None
    cached_angles = np.asarray(cached["steering_angles"], dtype=np.float64)
    if cached_angles.shape != steering_angles.shape or not np.allclose(cached_angles, steering_angles):
        return None
    return {
        "scan_lines": np.asarray(cached["scan_lines"], dtype=np.float64),
        "runtime": float(cached["runtime"]),
    }


def _save_scan_line_cache(
    path: os.PathLike[str],
    scan_lines: np.ndarray,
    steering_angles: np.ndarray,
    nt: int,
    seed: int,
    runtime: float,
) -> None:
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        scan_lines=np.asarray(scan_lines, dtype=np.float64),
        steering_angles=np.asarray(steering_angles, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
        runtime=np.array(runtime, dtype=np.float64),
    )


# ── Helper: coordinate extraction (mirrors full compare script) ───────────────

def matlab_ordered_active_voxel_coords(not_transducer):
    active_mask = np.asarray(not_transducer.active_elements_mask)
    linear_indices = matlab_find(active_mask).astype(int).ravel() - 1
    coords = np.column_stack(np.unravel_index(linear_indices, active_mask.shape, order="F"))
    return coords.astype(int)


def c_order_active_voxel_coords(not_transducer):
    active_mask = np.asarray(not_transducer.active_elements_mask)
    return np.argwhere(active_mask != 0).astype(int)


def build_phased_source_mask(active_coords, fnx, fny, fnz, px, py, pz):
    mask = np.zeros((fnx, fny, fnz), dtype=np.float64)
    for x_idx, y_idx, z_idx in active_coords:
        mask[x_idx + px, y_idx + py, z_idx + pz] = 1.0
    return mask


def build_phased_velocity_signals(kgrid, not_transducer, input_signal, active_coords):
    delay_mask = np.asarray(not_transducer.delay_mask(), dtype=int)
    transmit_apod_mask = np.asarray(not_transducer.transmit_apodization_mask, dtype=np.float64)
    transducer_signal = np.asarray(not_transducer.input_signal, dtype=np.float64).reshape(-1)
    # kwavers applies 2*c0*dt/dx internally for additive velocity sources
    # (commit caabc640). Do NOT apply the factor here.

    nt = int(kgrid.Nt)
    ux_signals = np.zeros((active_coords.shape[0], nt), dtype=np.float64)
    for point, (x_idx, y_idx, z_idx) in enumerate(active_coords):
        delay = int(delay_mask[x_idx, y_idx, z_idx])
        weight = float(transmit_apod_mask[x_idx, y_idx, z_idx])
        start = delay
        end = min(delay + nt, transducer_signal.size)
        if end > start:
            ux_signals[point, : end - start] = transducer_signal[start:end] * weight
    return ux_signals, 1.0


def normalize_sensor_matrix(raw, expected_sensors: int):
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[0] != expected_sensors:
        if raw.shape[1] == expected_sensors:
            raw = raw.T
        else:
            raise ValueError(f"sensor shape {raw.shape} vs expected {expected_sensors} rows")
    return raw


def build_reference_objects(seed: int):
    rng = np.random.default_rng(seed)

    kgrid = kWaveGrid(ACTIVE_GRID, GRID_SPACING_METERS)
    t_end = (ACTIVE_GRID.x * GRID_SPACING_METERS.x) * 2.2 / C0
    kgrid.makeTime(C0, t_end=t_end)

    medium = kWaveMedium(sound_speed=None, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER, BonA=BON_A)

    input_signal_raw = tone_burst(1 / kgrid.dt, TONE_BURST_FREQ, TONE_BURST_CYCLES)
    input_signal = (SOURCE_STRENGTH / (C0 * RHO0)) * input_signal_raw

    transducer_width = N_ELEMENTS * ELEMENT_WIDTH + (N_ELEMENTS - 1) * ELEMENT_SPACING
    t_pos = dotdict()
    t_pos.number_elements = N_ELEMENTS
    t_pos.element_width = ELEMENT_WIDTH
    t_pos.element_length = ELEMENT_LENGTH
    t_pos.element_spacing = ELEMENT_SPACING
    t_pos.radius = float("inf")
    t_pos.position = np.round([
        1,
        ACTIVE_GRID.y / 2 - transducer_width / 2,
        ACTIVE_GRID.z / 2 - ELEMENT_LENGTH / 2,
    ])
    transducer = kWaveTransducerSimple(kgrid, **t_pos)

    nt_params = dotdict()
    nt_params.sound_speed = C0
    nt_params.focus_distance = 20e-3
    nt_params.elevation_focus_distance = 20e-3
    nt_params.steering_angle = 0
    nt_params.steering_angle_max = 16
    nt_params.transmit_apodization = "Rectangular"
    nt_params.receive_apodization = "Rectangular"
    nt_params.active_elements = np.ones((N_ELEMENTS, 1))
    nt_params.input_signal = input_signal
    not_transducer = NotATransducer(transducer, kgrid, **nt_params)

    # Homogeneous medium (no scatterers) for speed
    sound_speed_map = np.full((kgrid.Nx, kgrid.Ny, kgrid.Nz), C0)
    density_map = np.full((kgrid.Nx, kgrid.Ny, kgrid.Nz), RHO0)
    medium.sound_speed = sound_speed_map
    medium.density = density_map
    return kgrid, medium, transducer, not_transducer, np.asarray(input_signal).ravel()


def run_kwave(kgrid, medium, not_transducer, steering_angles, use_gpu: bool = False):
    scan_lines = np.zeros((len(steering_angles), kgrid.Nt))
    for ai, angle in enumerate(steering_angles):
        print(f"  [kwave] angle {ai+1}/{len(steering_angles)} = {angle}°")
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
        scan_lines[ai, :] = not_transducer.scan_line(
            not_transducer.combine_sensor_data(sensor_data["p"].T)
        )
    return scan_lines


def run_pykwavers(kgrid, medium_kw, transducer, not_transducer, input_signal, steering_angles, use_gpu: bool):
    fnx, fny, fnz = int(TOTAL_GRID.x), int(TOTAL_GRID.y), int(TOTAL_GRID.z)
    px, py, pz = int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)
    grid = pkw.Grid(fnx, fny, fnz, GRID_SPACING_METERS.x, GRID_SPACING_METERS.y, GRID_SPACING_METERS.z)

    ss_full = np.full((fnx, fny, fnz), C0, dtype=np.float64)
    rho_full = np.full((fnx, fny, fnz), RHO0, dtype=np.float64)
    ss_full[px:px+ACTIVE_GRID.x, py:py+ACTIVE_GRID.y, pz:pz+ACTIVE_GRID.z] = np.asarray(medium_kw.sound_speed)
    rho_full[px:px+ACTIVE_GRID.x, py:py+ACTIVE_GRID.y, pz:pz+ACTIVE_GRID.z] = np.asarray(medium_kw.density)
    absorb_full = np.full((fnx, fny, fnz), ALPHA_COEFF, dtype=np.float64)
    bona_full = np.full((fnx, fny, fnz), BON_A, dtype=np.float64)

    active_coords = c_order_active_voxel_coords(not_transducer)
    base_mask = build_phased_source_mask(active_coords, fnx, fny, fnz, px, py, pz)
    expected_sensors = N_ELEMENTS * ELEMENT_WIDTH * ELEMENT_LENGTH

    if use_gpu:
        session = pkw.GpuPstdSession(
            grid, ss_full, rho_full,
            dt=kgrid.dt, time_steps=kgrid.Nt,
            absorption=absorb_full, nonlinearity=bona_full,
            pml_size_xyz=(px, py, pz), alpha_power=ALPHA_POWER,
        )
        session.set_source_sensor_mask(base_mask)

    scan_lines = np.zeros((len(steering_angles), kgrid.Nt))
    for ai, angle in enumerate(steering_angles):
        print(f"  [pkw {'GPU' if use_gpu else 'CPU'}] angle {ai+1}/{len(steering_angles)} = {angle}°")
        not_transducer.steering_angle = float(angle)
        ux_signals, transducer_scale = build_phased_velocity_signals(
            kgrid, not_transducer, input_signal, active_coords,
        )
        print(f"    drive: n_vox={ux_signals.shape[0]} max={np.max(np.abs(ux_signals)):.4g} scale={transducer_scale:.4g}")

        if use_gpu:
            session.set_velocity_signals(ux_signals)
            raw = normalize_sensor_matrix(
                np.asarray(session.run_scan_line_cached()), expected_sensors
            )
        else:
            pkw_med = pkw.Medium(
                sound_speed=ss_full, density=rho_full,
                absorption=absorb_full, nonlinearity=bona_full,
            )
            src = pkw.Source.from_velocity_mask_2d(base_mask, ux=ux_signals, mode="additive")
            sen = pkw.Sensor.from_mask(base_mask.astype(bool))
            sim = pkw.Simulation(grid, pkw_med, src, sen, solver=pkw.SolverType.PSTD)
            sim.set_pml_size_xyz(px, py, pz)
            sim.set_nonlinear(True)
            sim.set_alpha_coeff(ALPHA_COEFF)
            sim.set_alpha_power(ALPHA_POWER)
            result = sim.run(kgrid.Nt, dt=kgrid.dt)
            raw = normalize_sensor_matrix(np.asarray(result.sensor_data), expected_sensors)

        scan_lines[ai, :] = not_transducer.scan_line(not_transducer.combine_sensor_data(raw))
    return scan_lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260401)
    ap.add_argument("--pykwavers-gpu", action="store_true")
    ap.add_argument("--kwave-gpu", action="store_true",
                    help="Use CUDA binary for k-wave-python (requires GPU + CUDA driver).")
    ap.add_argument("--nonlinear", action="store_true",
                    help="Enable nonlinearity in pykwavers leg (always on in kwave).")
    ap.add_argument("--angles", type=str, default=None,
                    help="Comma-separated steering angles override, e.g. '0' or '-8,0,8'.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Force re-run both legs, ignoring cached NPZ files.")
    ap.add_argument("--allow-failure", action="store_true",
                    help="Exit 0 even when parity targets are not met.")
    args = ap.parse_args()

    if args.no_cache:
        global REFRESH_CACHE
        REFRESH_CACHE = True

    global STEERING_ANGLES
    if args.angles is not None:
        STEERING_ANGLES = np.array([float(a) for a in args.angles.split(",")])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Tiny grid: {TOTAL_GRID.x}x{TOTAL_GRID.y}x{TOTAL_GRID.z} PML={PML_SIZE.x},{PML_SIZE.y},{PML_SIZE.z} "
          f"active={ACTIVE_GRID.x}x{ACTIVE_GRID.y}x{ACTIVE_GRID.z}")
    print(f"Transducer: {N_ELEMENTS} elem x (w={ELEMENT_WIDTH} x l={ELEMENT_LENGTH}) = {N_ELEMENTS*ELEMENT_WIDTH*ELEMENT_LENGTH} voxels")

    kgrid, medium, transducer, not_transducer, input_signal = build_reference_objects(args.seed)
    print(f"Grid: nt={kgrid.Nt}  dt={kgrid.dt:.3e}  dx={kgrid.dx:.3e}")

    # ── k-wave reference ──────────────────────────────────────────────────────
    _kw_cached = _load_scan_line_cache(KWAVE_CACHE, STEERING_ANGLES, kgrid.Nt, args.seed)
    if _kw_cached is not None:
        kw_lines = _kw_cached["scan_lines"]
        kw_time = _kw_cached["runtime"]
        print(f"k-wave loaded from cache ({kw_time:.1f}s original runtime)")
    else:
        t0 = time.perf_counter()
        kw_lines = run_kwave(kgrid, medium, not_transducer, STEERING_ANGLES, use_gpu=args.kwave_gpu)
        kw_time = time.perf_counter() - t0
        print(f"k-wave finished in {kw_time:.1f}s")
        _save_scan_line_cache(KWAVE_CACHE, kw_lines, STEERING_ANGLES, kgrid.Nt, args.seed, kw_time)

    # ── pykwavers leg ─────────────────────────────────────────────────────────
    _pkw_cached = _load_scan_line_cache(PYKWAVERS_CACHE, STEERING_ANGLES, kgrid.Nt, args.seed)
    if _pkw_cached is not None:
        pkw_lines = _pkw_cached["scan_lines"]
        pkw_time = _pkw_cached["runtime"]
        print(f"pykwavers loaded from cache ({pkw_time:.1f}s original runtime)")
    else:
        t0 = time.perf_counter()
        pkw_lines = run_pykwavers(
            kgrid, medium, transducer, not_transducer, input_signal,
            STEERING_ANGLES, use_gpu=args.pykwavers_gpu,
        )
        pkw_time = time.perf_counter() - t0
        print(f"pykwavers finished in {pkw_time:.1f}s")
        _save_scan_line_cache(PYKWAVERS_CACHE, pkw_lines, STEERING_ANGLES, kgrid.Nt, args.seed, pkw_time)

    # ── Metrics per scan line ─────────────────────────────────────────────────
    rms_ratios, peak_ratios, pearson_rs = [], [], []
    for ai, angle in enumerate(STEERING_ANGLES):
        m = compute_trace_metrics(kw_lines[ai], pkw_lines[ai])
        rms_ratios.append(m["rms_ratio"])
        peak_ratios.append(m["peak_ratio"])
        pearson_rs.append(m["pearson_r"])
        print(f"  angle={angle:+.0f}°  r={m['pearson_r']:.4f}  rms_ratio={m['rms_ratio']:.4f}  "
              f"peak_ratio={m['peak_ratio']:.4f}  kw_peak={m['reference_peak']:.3g}  pkw_peak={m['candidate_peak']:.3g}")

    mean_rms = float(np.mean(rms_ratios))
    mean_r = float(np.mean(pearson_rs))
    print(f"\nMean rms_ratio={mean_rms:.4f}  mean_pearson_r={mean_r:.4f}")

    # ── Image-level metrics ───────────────────────────────────────────────────
    img_m = compute_image_metrics(kw_lines, pkw_lines)
    label = "pykwavers GPU" if args.pykwavers_gpu else "pykwavers CPU"

    report = [
        "us_bmode_phased_array_tiny parity metrics",
        f"grid: {TOTAL_GRID.x}x{TOTAL_GRID.y}x{TOTAL_GRID.z}, PML={PML_SIZE.x}",
        f"transducer: {N_ELEMENTS} elements x {ELEMENT_WIDTH}w x {ELEMENT_LENGTH}l",
        f"angles: {[int(angle) for angle in STEERING_ANGLES]}",
        f"solver: {label}",
        "",
        "Per-angle (kwave vs pkw scan-line trace):",
    ]
    for ai, angle in enumerate(STEERING_ANGLES):
        report.append(f"  {float(angle):+.1f}°  r={pearson_rs[ai]:.4f}  rms_ratio={rms_ratios[ai]:.4f}")
    report += [
        "",
        f"Mean rms_ratio : {mean_rms:.6f}",
        f"Mean pearson_r : {mean_r:.6f}",
        f"Image pearson_r: {img_m['pearson_r']:.6f}",
        f"Image PSNR dB  : {img_m['psnr_db']:.6f}",
        f"Image RMS ratio: {img_m['rms_ratio']:.6f}",
        "",
        f"k-wave time    : {kw_time:.1f}s",
        f"pykwavers time : {pkw_time:.1f}s",
    ]
    overall_status = (
        "PASS"
        if mean_r >= PARITY_THRESHOLDS["mean_pearson_r"]
        and PARITY_THRESHOLDS["mean_rms_ratio_min"]
        <= mean_rms
        <= PARITY_THRESHOLDS["mean_rms_ratio_max"]
        else "FAIL"
    )
    report.append(f"parity_status: {overall_status}")
    save_text_report(METRICS_PATH, "us_bmode_phased_array_tiny metrics", report)
    print(f"Status: {overall_status}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(STEERING_ANGLES), 1, figsize=(12, 3*len(STEERING_ANGLES)))
    if len(STEERING_ANGLES) == 1:
        axes = [axes]
    t = np.arange(kgrid.Nt) * kgrid.dt * 1e6
    for ai, (ax, angle) in enumerate(zip(axes, STEERING_ANGLES)):
        ax.plot(t, kw_lines[ai], label="k-wave", color="C0", lw=1.5)
        ax.plot(t, pkw_lines[ai], label=label, color="C3", lw=1.1, ls="--")
        ax.set_title(f"angle={angle:+.0f}°  r={pearson_rs[ai]:.3f}  rms_ratio={rms_ratios[ai]:.3f}")
        ax.set_xlabel("Time [µs]")
        ax.set_ylabel("Pressure")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, ls=":")
    fig.suptitle(f"Tiny phased-array  mean_rms_ratio={mean_rms:.3f}  mean_r={mean_r:.3f}", y=1.01)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {PNG_PATH}")
    print(f"Saved: {METRICS_PATH}")

    return 0 if overall_status == "PASS" or args.allow_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
