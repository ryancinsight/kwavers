#!/usr/bin/env python3
"""
Comparison: pykwavers vs k-wave-python for Transducer Definition

This script reimplements the 'us_defining_transducer' k-wave example natively in
pykwavers using TransducerArray2D, and subsequently executes the exact same
configuration in the k-wave-python wrapped 'kspaceFirstOrder3D'.

It compares the central output pressures from both engines at 3 sensor positions.

Author: Ryan Clanton (@ryancinsight)
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import pykwavers as kw
    from pykwavers.comparison import compute_error_metrics
except ImportError:
    print("[X] pykwavers not available. Build with: maturin develop --release")
    sys.exit(1)

try:
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
    from kwave.kWaveSimulation import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.utils.dotdictionary import dotdict
    from kwave.utils.filters import spect
    from kwave.utils.signals import tone_burst
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False
    print("WARNING: k-wave-python not installed. Only pykwavers base runs will execute.")


# ============================================================================
# Shared Configuration
# ============================================================================

# Define Grid & Medium
PML_X_SIZE = 20
PML_Y_SIZE = 10
PML_Z_SIZE = 10
Nx = 128 - 2 * PML_X_SIZE
Ny = 128 - 2 * PML_Y_SIZE
Nz = 64 - 2 * PML_Z_SIZE
x_len = 40e-3
dx = x_len / 128
dy = dx
dz = dx

sound_speed = 1540.0
density = 1000.0

t_end = 40e-6
# dt = CFL * dx / c
# Not hardcoding dt yet, we map from kgrid if available or use pykwavers dt

source_strength = 1e6 # Pa, wait, particle velocity if divided by Z
tone_burst_freq = 0.5e6
tone_burst_cycles = 5

def run_kwave_reference():
    """Runs the exact us_defining_transducer setup through k-wave-python."""
    print("=" * 80)
    print("Running k-wave-python reference simulation")
    print("=" * 80)
    
    start_time = time.perf_counter()
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
    
    # define the medium WITHOUT absorption to ensure clean 1:1 mathematical parity first,
    # or keep exactly as original: alpha_coeff=0.75, alpha_power=1.5, BonA=6
    # For now, keep original to test parity.
    medium = kWaveMedium(sound_speed=sound_speed, density=density, alpha_coeff=0, alpha_power=1.5, BonA=6)
    
    kgrid.makeTime(medium.sound_speed, t_end=t_end)
    
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
    input_signal = (source_strength / (medium.sound_speed * medium.density)) * input_signal
    
    transducer = dotdict()
    transducer.number_elements = 72
    transducer.element_width = 1
    transducer.element_length = 12
    transducer.element_spacing = 0
    transducer.radius = np.inf
    transducer.position = np.round([1, Ny / 2 - Nx // 2, Nz / 2 - transducer.element_length / 2])
    transducer = kWaveTransducerSimple(kgrid, **transducer)
    
    not_transducer = dotdict()
    not_transducer.sound_speed = medium.sound_speed
    not_transducer.focus_distance = 20e-3
    not_transducer.elevation_focus_distance = 19e-3
    not_transducer.steering_angle = 0
    not_transducer.transmit_apodization = "Rectangular"
    not_transducer.receive_apodization = "Rectangular"
    not_transducer.active_elements = np.zeros((transducer.number_elements, 1))
    not_transducer.active_elements[21:52] = 1 # 31 elements active
    not_transducer.input_signal = input_signal
    
    not_transducer = NotATransducer(transducer, kgrid, **not_transducer)
    
    sensor_mask = np.zeros((Nx, Ny, Nz))
    sensor_mask[Nx // 4, Ny // 2, Nz // 2] = 1
    sensor_mask[Nx // 2, Ny // 2, Nz // 2] = 1
    sensor_mask[3 * Nx // 4, Ny // 2, Nz // 2] = 1
    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]
    
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=Vector([PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE]),
        data_cast="single",
        save_to_disk=True,
    )
    
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    
    sensor_data = kspaceFirstOrder3D(
        medium=medium,
        kgrid=kgrid,
        source=not_transducer,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    
    # Extract actual mask computed by k-wave
    kw_ux = None
    if hasattr(not_transducer, 'ux'):
        kw_ux = not_transducer.ux
    
    runtime = time.perf_counter() - start_time
    print(f"k-wave completed in {runtime:.2f}s")
    
    kw_res = {
        "pressure": sensor_data["p"],
        "time": kgrid.t_array,
        "dt": kgrid.dt,
        "input_signal": not_transducer.input_signal,
    }
    np.savez("kw_results.npz", 
             pressure=kw_res["pressure"], 
             time=kw_res["time"], 
             dt=kw_res["dt"], 
             input_signal=kw_res["input_signal"],
             ux=kw_ux)
    return kw_res


def run_pykwavers(kw_dt, input_signal):
    """Runs the equivalent setup in PyKwavers native API with k-wave-matching
    beamforming delays and PML configuration."""
    print("=" * 80)
    print("Running pykwavers native simulation")
    print("=" * 80)
    
    start_time = time.perf_counter()
    
    print("1. Initializing PyKwavers Grid (128×128×64 matching k-wave total grid)...", flush=True)
    # k-wave with pml_inside=False uses a total grid of 128×128×64 where the PML
    # wraps the inner 88×108×44 region externally. To produce identical spectral
    # domain behavior, we use the same total grid with pml_inside=True.
    Gx = 128
    Gy = 128
    Gz = 64
    grid = kw.Grid(nx=Gx, ny=Gy, nz=Gz, dx=dx, dy=dy, dz=dz)
    
    print("2. Initializing Medium Properties...", flush=True)
    medium = kw.Medium.homogeneous(sound_speed=sound_speed, density=density)
    
    print("3. Computing NotATransducer-equivalent beamforming delays...", flush=True)
    
    # --- Transducer geometry parameters ---
    n_elements = 72
    element_width = 1       # grid points
    element_length = 12     # grid points  
    element_spacing = 0     # grid points between elements
    focus_distance = 20e-3  # [m]
    elevation_focus_distance = 19e-3  # [m]
    steering_angle = 0      # [deg]
    
    # Active elements: indices 21 through 51 inclusive (31 active)
    active_mask = np.zeros(n_elements, dtype=bool)
    active_mask[21:52] = True
    n_active = int(active_mask.sum())  # 31
    
    # Element pitch = (element_width + element_spacing) * dx
    element_pitch = (element_width + element_spacing) * dx  # [m]
    
    # --- Azimuth beamforming delays (per active element) ---
    # k-wave: element_index centered around 0
    active_indices = np.where(active_mask)[0]
    element_index_az = np.arange(-(n_active - 1) / 2, (n_active + 1) / 2)
    
    if np.isinf(focus_distance):
        az_delays_s = element_pitch * element_index_az * np.sin(np.radians(steering_angle)) / sound_speed
    else:
        az_delays_s = (focus_distance / sound_speed) * (
            1 - np.sqrt(
                1
                + (element_index_az * element_pitch / focus_distance) ** 2
                - 2 * (element_index_az * element_pitch / focus_distance) * np.sin(np.radians(steering_angle))
            )
        )
    
    # Convert to integer sample delays and NEGATE (k-wave reverses)
    az_delays = -np.round(az_delays_s / kw_dt).astype(int)  # [samples]
    
    # --- Elevation beamforming delays (per voxel within element) ---
    # element_length = 12 grid points in Z
    elev_index = np.arange(-(element_length - 1) / 2, (element_length + 1) / 2)
    
    if not np.isinf(elevation_focus_distance):
        elev_delays_s = (elevation_focus_distance - np.sqrt(
            (elev_index * dz) ** 2 + elevation_focus_distance ** 2
        )) / sound_speed
        elev_delays = -np.round(elev_delays_s / kw_dt).astype(int)  # [samples]
    else:
        elev_delays = np.zeros(element_length, dtype=int)
    
    # --- Build per-voxel total delay ---
    total_delays = np.zeros(n_active * element_length, dtype=int)
    for el_idx in range(n_active):
        for ev_idx in range(element_length):
            voxel_idx = el_idx * element_length + ev_idx
            total_delays[voxel_idx] = az_delays[el_idx] + elev_delays[ev_idx]
    
    # Offset delays so minimum is 0 (k-wave does this)
    total_delays -= total_delays.min()
    max_delay = total_delays.max()
    
    print(f"   --> Beamforming: {n_active} elements x {element_length} elevation = {n_active * element_length} source pts")
    print(f"   --> Delay range: 0 to {max_delay} samples ({max_delay * kw_dt * 1e6:.2f} µs)")
    
    # --- Build source mask and per-point signals ---
    # Source positions in the FULL grid are offset by PML thickness
    # k-wave places transducer at [1, Ny/2 - Nx//2, Nz/2 - 12/2] in the inner grid
    # In the full grid, add PML offsets: x += PML_X, y += PML_Y, z += PML_Z
    pml_offset_x = PML_X_SIZE
    pml_offset_y = PML_Y_SIZE
    pml_offset_z = PML_Z_SIZE
    
    offset_x = 0 + pml_offset_x
    offset_y = int(Ny / 2 - Nx // 2) - 1 + pml_offset_y
    offset_z = int(Nz / 2 - element_length // 2) - 1 + pml_offset_z
    
    num_time_steps = int(t_end / kw_dt)
    flat_signal = input_signal.flatten()
    
    # Build the mask on the FULL grid
    source_mask = np.zeros((Gx, Gy, Gz))
    source_points = []
    
    for el_idx in range(n_active):
        global_el = active_indices[el_idx]
        for ev_idx in range(element_length):
            iy = offset_y + global_el
            iz = offset_z + ev_idx
            source_mask[offset_x, iy, iz] = 1.0
            voxel_idx = el_idx * element_length + ev_idx
            source_points.append((offset_x, iy, iz, total_delays[voxel_idx]))
    
    # Build per-source-point delayed velocity signals
    n_source_pts = int(source_mask.sum())
    total_sig_len = num_time_steps
    ux_2d = np.zeros((n_source_pts, total_sig_len), dtype=flat_signal.dtype)
    
    for pt_idx, (_, _, _, delay) in enumerate(source_points):
        start = delay
        end = min(start + len(flat_signal), total_sig_len)
        sig_len = end - start
        if sig_len > 0:
            ux_2d[pt_idx, start:end] = flat_signal[:sig_len]
    
    source = kw.Source.from_velocity_mask_2d(source_mask, ux=ux_2d, mode='dirichlet')
    print(f"   --> Source mask generated (velocity-x Dirichlet, per-voxel delayed). n_source_pts={n_source_pts}")
    
    print("4. Placing Sensors (with PML offset)...", flush=True)
    sensor_mask = np.zeros((Gx, Gy, Gz), dtype=bool)
    # Sensor positions in full grid: inner_pos + pml_offset
    sensor_mask[Nx // 4 + pml_offset_x, Ny // 2 + pml_offset_y, Nz // 2 + pml_offset_z] = True
    sensor_mask[Nx // 2 + pml_offset_x, Ny // 2 + pml_offset_y, Nz // 2 + pml_offset_z] = True
    sensor_mask[3 * Nx // 4 + pml_offset_x, Ny // 2 + pml_offset_y, Nz // 2 + pml_offset_z] = True
    
    sensor = kw.Sensor.from_mask(sensor_mask)
    
    print("5. Launching PSTD Simulation Backend...", flush=True)
    # Use full grid with PML inside, matching k-wave's effective domain
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    sim.set_pml_size_xyz(PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE)  # k-wave [20, 10, 10]
    sim.set_pml_inside(True)  # PML inside our full 128x128x64 grid
    
    print(f"   --> Grid: {Gx}×{Gy}×{Gz}, PML size: {sim.pml_size}, run loop: {num_time_steps} steps...", flush=True)
    result = sim.run(time_steps=num_time_steps, dt=kw_dt)
    
    runtime = time.perf_counter() - start_time
    print(f"pykwavers completed in {runtime:.2f}s")
    
    return {
        "pressure": result.sensor_data,
        "time": result.time,
        "runtime": runtime,
    }


def main():
    print("Phase 14: US Transducer PyKwavers vs k-wave-python Comparison\n")
    if not KWAVE_AVAILABLE:
        print("Cannot run comparison: kwave-python not installed.")
        return 1

    if Path("kw_results.npz").exists():
        print("Using cached k-wave-python reference simulation data...")
        data = np.load("kw_results.npz")
        kw_results = {"pressure": data["pressure"], "time": data["time"], "dt": data["dt"], "input_signal": data["input_signal"]}
    else:
        kw_results = run_kwave_reference()
    dt = float(kw_results["dt"])
    input_signal = kw_results["input_signal"]
    
    py_results = run_pykwavers(dt, input_signal)
    
    # Verify Metrics
    kw_p_raw = kw_results["pressure"]
    py_p_raw = py_results["pressure"]
    
    print(f"\n[DEBUG] kw_p_raw shape: {kw_p_raw.shape}")
    print(f"[DEBUG] py_p_raw shape: {py_p_raw.shape}")
    
    # k-wave returns (time_steps, n_sensors), PyKwavers returns (n_sensors, time_steps)
    # Normalize both to (n_sensors, time_steps) for comparison
    if kw_p_raw.ndim == 2 and kw_p_raw.shape[1] < kw_p_raw.shape[0]:
        kw_p = kw_p_raw.T  # (sensors, time)
    else:
        kw_p = kw_p_raw
    
    if py_p_raw.ndim == 1:
        py_p = py_p_raw.reshape(1, -1)  # single sensor → (1, time)
    else:
        py_p = py_p_raw
    
    n_sensors = min(kw_p.shape[0], py_p.shape[0])
    n_common_ts = min(kw_p.shape[1], py_p.shape[1])
    print(f"[DEBUG] Aligned: {n_sensors} sensors x {n_common_ts} time steps")
    
    print(f"[DEBUG] py_p bounds: Max={np.max(py_p):.4e}, Min={np.min(py_p):.4e}, NaNs={np.isnan(py_p).sum()}")
    print(f"[DEBUG] kw_p bounds: Max={np.max(kw_p):.4e}, Min={np.min(kw_p):.4e}, NaNs={np.isnan(kw_p).sum()}")
    
    print("\nMetrics:")
    for i in range(n_sensors):
        kw_trace = kw_p[i, :n_common_ts]
        py_trace = py_p[i, :n_common_ts]
        metrics = compute_error_metrics(kw_trace, py_trace)
        
        # Find peak arrival times
        kw_peak_idx = np.argmax(np.abs(kw_trace))
        py_peak_idx = np.argmax(np.abs(py_trace))
        kw_peak_t = kw_peak_idx * dt * 1e6  # µs
        py_peak_t = py_peak_idx * dt * 1e6
        kw_peak_amp = kw_trace[kw_peak_idx]
        py_peak_amp = py_trace[py_peak_idx]
        amp_ratio = abs(py_peak_amp / kw_peak_amp) if abs(kw_peak_amp) > 1e-10 else float('inf')
        
        print(f" Sensor {i+1}:")
        print(f"  - L2 Error: {metrics['l2_error']:.4f}")
        print(f"  - L-inf Error: {metrics['linf_error']:.4f}")
        print(f"  - Correlation: {metrics['correlation']:.4f}")
        print(f"  - Peak time: kw={kw_peak_t:.2f}µs, py={py_peak_t:.2f}µs, Δ={py_peak_t-kw_peak_t:.2f}µs")
        print(f"  - Peak amp: kw={kw_peak_amp:.2e}, py={py_peak_amp:.2e}, ratio={amp_ratio:.3f}")
        
    # Plotting
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    offset = -30e5
    for i in range(n_sensors):
        t_kw = kw_results["time"].squeeze()[:kw_p.shape[1]] * 1e6
        axes[0].plot(t_kw, kw_p[i, :] + offset * i, 'k-', label=f"kW S{i+1}" if i==0 else "")
        
        t_py = py_results["time"].squeeze()[:py_p.shape[1]] * 1e6
        axes[0].plot(t_py, py_p[i, :] + offset * i, 'r--', label=f"pyk S{i+1}" if i==0 else "")

    axes[0].set_yticks([offset * i for i in range(3)], ["Sensor 1", "Sensor 2", "Sensor 3"])
    axes[0].set_xlabel("Time [\u03BCs]")
    axes[0].set_title("Time-Domain Pressure Comparison")
    axes[0].legend()

    # Spectra
    axes[1].set_title("Frequency Spectra Comparison")
    axes[1].set_xlabel("Frequency [MHz]")
    f_max = sound_speed / (2 * dx)
    axes[1].set_xlim([0, f_max * 1e-6])
    
    plt.tight_layout()
    plt.savefig("transducer_comparison.png")
    print("\nSaved transducer_comparison.png")

if __name__ == "__main__":
    sys.exit(main())
