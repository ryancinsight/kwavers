import os
import sys
import numpy as np
import time

# Try importing pykwavers and kwave
try:
    import pykwavers as kw
except ImportError:
    print("Could not import pykwavers. Make sure you run `maturin develop` inside the python virtual environment.")
    sys.exit(1)
    
try:
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    from kwave.utils.signals import tone_burst
    from kwave.utils.dotdictionary import dotdict
except ImportError:
    print("Could not import k-wave-python. Make sure it is installed.")
    sys.exit(1)


def compute_error_metrics(kw_trace, py_trace):
    ref_norm = np.linalg.norm(kw_trace)
    if ref_norm < 1e-12:
        return {'l2_error': 0.0, 'linf_error': 0.0, 'correlation': 0.0}
    
    l2_error = np.linalg.norm(kw_trace - py_trace) / ref_norm
    
    ref_inf = np.max(np.abs(kw_trace))
    if ref_inf < 1e-12:
        linf_error = 0.0
    else:
        linf_error = np.max(np.abs(kw_trace - py_trace)) / ref_inf
        
    if np.std(kw_trace) < 1e-12 or np.std(py_trace) < 1e-12:
        correlation = 0.0
    else:
        correlation = np.corrcoef(kw_trace, py_trace)[0, 1]
        
    return {
        'l2_error': l2_error,
        'linf_error': linf_error,
        'correlation': correlation
    }

def main():
    # Grid size (larger grid to prevent boundary reflections)
    Nx = 128
    Ny = 96
    Nz = 96
    dx = 1e-3
    dy = dx
    dz = dx

    PML_SIZE = 16
    
    sound_speed = 1500.0
    density = 1000.0
    
    # Time settings
    # To capture a few waves passing through
    # Distance from center to edge is ~32 * 1e-3 = 3.2cm. 
    # Time to travel 3.2cm is 3.2e-2 / 1500 = 21 microseconds
    t_end = 25e-6
    
    print("=" * 80)
    print("Running k-wave-python reference simulation")
    print("=" * 80)
    
    # CANONICAL: same total grid as kwavers (do NOT subtract 2*PML_SIZE)
    # pml_inside=True means PML is counted within this Nx, same as kwavers
    kgrid = kWaveGrid(
        [Nx, Ny, Nz],
        [dx, dy, dz]
    )
    
    medium = kWaveMedium(
        sound_speed=sound_speed, 
        density=density, 
        alpha_coeff=0.0,
        alpha_power=1.5
    )
    
    kgrid.makeTime(medium.sound_speed, t_end=t_end)
    kw_dt = kgrid.dt
    print(f"k-wave dt: {kw_dt:.4e} s")
    print(f"k-wave steps: {kgrid.Nt}")
    
    # Point source in the middle of the inner grid
    source_mask = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz))
    source_mask[kgrid.Nx//2, kgrid.Ny//2, kgrid.Nz//2] = 1
    
    tone_burst_freq = 0.5e6
    tone_burst_cycles = 3
    input_signal = tone_burst(1 / kw_dt, tone_burst_freq, tone_burst_cycles).flatten()
    
    print("\n[DEBUG] RAW INPUT SIGNAL DUMP (first 30 steps):")
    for t_i in range(min(30, len(input_signal))):
        print(f" t={t_i:02d} | source: {input_signal[t_i]:+10.4e}")

    if len(input_signal) < kgrid.Nt:
        input_signal = np.pad(input_signal, (0, kgrid.Nt - len(input_signal)))
    elif len(input_signal) > kgrid.Nt:
        input_signal = input_signal[:kgrid.Nt]
        
    input_signal_2d = input_signal.reshape(1, -1)
    
    source = kSource()
    source.p_mask = source_mask
    source.p = input_signal_2d
    source.p_mode = "additive"
    
    # 3 sensors at different distances along x axis
    sensor_mask = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz))
    sensor_mask[kgrid.Nx//2 + 4, kgrid.Ny//2, kgrid.Nz//2] = 1   # Near
    sensor_mask[kgrid.Nx//2 + 10, kgrid.Ny//2, kgrid.Nz//2] = 1  # Mid
    sensor_mask[kgrid.Nx//2 + 18, kgrid.Ny//2, kgrid.Nz//2] = 1  # Far
    
    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]
    
    simulation_options = SimulationOptions(
        pml_inside=True,
        pml_size=PML_SIZE,
        data_cast="single",
        save_to_disk=True,
    )
    
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    
    start_time = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    kw_time = time.perf_counter() - start_time
    print(f"k-wave completed in {kw_time:.2f}s")
    
    kw_p = sensor_data["p"]
    
    print("=" * 80)
    print("Running pykwavers native simulation")
    print("=" * 80)
    
    # Total grid includes PML
    py_grid = kw.Grid(Nx, Ny, Nz, dx, dy, dz)
    py_medium = kw.Medium.homogeneous(sound_speed, density)
    
    py_source_mask = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    py_source_mask[Nx//2, Ny//2, Nz//2] = 1.0
    py_source = kw.Source.from_mask(py_source_mask, input_signal.flatten(), tone_burst_freq, mode="additive")
    
    py_sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    py_sensor_mask[Nx//2 + 4, Ny//2, Nz//2] = 1.0
    py_sensor_mask[Nx//2 + 10, Ny//2, Nz//2] = 1.0
    py_sensor_mask[Nx//2 + 18, Ny//2, Nz//2] = 1.0
    py_sensor = kw.Sensor.from_mask(py_sensor_mask)
    
    sim = kw.Simulation(py_grid, py_medium, py_source, py_sensor, solver=kw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)
    
    start_time = time.perf_counter()
    py_result = sim.run(time_steps=kgrid.Nt, dt=kw_dt)
    py_time = time.perf_counter() - start_time
    print(f"pykwavers completed in {py_time:.2f}s")
    
    py_p = py_result.sensor_data
    
    # k-wave: (time, sensors) -> (sensors, time)
    if kw_p.ndim == 2 and kw_p.shape[1] < kw_p.shape[0]:
        kw_p = kw_p.T

    # Flatten if 1 sensor
    if py_p.ndim == 1:
        py_p = py_p.reshape(1, -1)

    # Canonical timing alignment:
    # k-Wave records at t=0 (before step 1); kwavers records after each step.
    # kw_p[:, 1:] aligns with py_p[:, :-1].
    kw_p = kw_p[:, 1:]
    py_p = py_p[:, :-1]

    n_sensors = min(kw_p.shape[0], py_p.shape[0])
    n_common_ts = min(kw_p.shape[1], py_p.shape[1])
    
    print(f"\n[DEBUG] py_p bounds: Max={np.max(py_p):.4e}, Min={np.min(py_p):.4e}, NaNs={np.isnan(py_p).sum()}")
    print(f"[DEBUG] kw_p bounds: Max={np.max(kw_p):.4e}, Min={np.min(kw_p):.4e}, NaNs={np.isnan(kw_p).sum()}")
    
    print("\nMetrics:")
    for i in range(n_sensors):
        kw_trace = kw_p[i, :n_common_ts]
        py_trace = py_p[i, :n_common_ts]
        metrics = compute_error_metrics(kw_trace, py_trace)

        kw_peak_idx = np.argmax(np.abs(kw_trace))
        py_peak_idx = np.argmax(np.abs(py_trace))
        kw_peak_t = kw_peak_idx * kw_dt * 1e6
        py_peak_t = py_peak_idx * kw_dt * 1e6
        kw_peak_amp = kw_trace[kw_peak_idx]
        py_peak_amp = py_trace[py_peak_idx]
        amp_ratio = abs(py_peak_amp / kw_peak_amp) if abs(kw_peak_amp) > 1e-10 else float('inf')
        
        print(f" Sensor {i+1}:")
        print(f"  - L2 Error: {metrics['l2_error']:.4f}")
        print(f"  - Correlation: {metrics['correlation']:.4f}")
        print(f"  - Peak time: kw={kw_peak_t:.2f}µs, py={py_peak_t:.2f}µs, Δ={py_peak_t-kw_peak_t:.2f}µs")
        print(f"  - Peak amp: kw={kw_peak_amp:.2e}, py={py_peak_amp:.2e}, ratio={amp_ratio:.3f}")
        
        if i == 0:
            print("\n  [DEBUG] RAW TRACE DUMP FOR SENSOR 1 (indices 10 to 40 around the start of the pulse):")
            for t_i in range(10, 40):
                print(f"   t={t_i:02d} | kw: {kw_trace[t_i]:+10.4e} | py: {py_trace[t_i]:+10.4e} | diff: {(kw_trace[t_i] - py_trace[t_i]):+10.4e}")

    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(n_sensors, 1, figsize=(10, 8), sharex=True)
        t_axis = np.arange(n_common_ts) * kw_dt * 1e6
        for i in range(n_sensors):
            axs[i].plot(t_axis, kw_p[i, :n_common_ts], label='k-wave', linestyle='--')
            axs[i].plot(t_axis, py_p[i, :n_common_ts], label='pykwavers', alpha=0.7)
            axs[i].set_title(f'Sensor {i+1}')
            axs[i].set_ylabel('Pressure')
            axs[i].legend()
        axs[-1].set_xlabel('Time (µs)')
        plt.tight_layout()
        plt.savefig("point_source_comparison.png", dpi=150)
        print("\nSaved point_source_comparison.png")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
