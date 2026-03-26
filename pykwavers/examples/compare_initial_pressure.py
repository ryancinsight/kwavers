import numpy as np
import pykwavers as kw
import time
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D

def main():
    Nx = 64
    Ny = 32
    Nz = 32
    dx = 1e-3
    PML_SIZE = 0
    
    sound_speed = 1500.0
    density = 1000.0
    
    # 1. Define k-wave grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    
    medium = kWaveMedium(sound_speed=sound_speed, density=density, alpha_coeff=0.0, alpha_power=1.5)
    
    kgrid.makeTime(medium.sound_speed)
    kw_dt = kgrid.dt
    
    # Define an initial pressure distribution
    p0 = np.zeros((Nx, Ny, Nz), dtype=float)
    # A small ball in the center
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if abs(i - Nx//2) <= 2:
                    p0[i, j, k] = 1.0
                    
    source = kSource()
    source.p0 = p0
    
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_mask[Nx//2 + 6, Ny//2, Nz//2] = 1
    sensor_mask[Nx//2 + 12, Ny//2, Nz//2] = 1
    sensor_mask[Nx//2 + 18, Ny//2, Nz//2] = 1
    
    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]
    
    simulation_options = SimulationOptions(
        pml_x_alpha=1.5, pml_y_alpha=1.5, pml_z_alpha=1.5,
        pml_x_size=PML_SIZE, pml_y_size=PML_SIZE, pml_z_size=PML_SIZE,
        pml_inside=True,
        save_to_disk=True,
        smooth_p0=False,
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    
    print("=" * 80)
    print("Running k-wave-python reference simulation")
    print("=" * 80)
    
    start_time = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    kw_time = time.perf_counter() - start_time
    kw_p = sensor_data["p"]
    
    print("=" * 80)
    print("Running pykwavers native simulation")
    print("=" * 80)
    
    py_grid = kw.Grid(Nx, Ny, Nz, dx, dx, dx)
    py_medium = kw.Medium.homogeneous(sound_speed, density)
    
    py_source = kw.Source.from_initial_pressure(p0)
    py_sensor = kw.Sensor.from_mask(sensor_mask)
    
    sim = kw.Simulation(py_grid, py_medium, py_source, py_sensor, solver=kw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)
    
    start_time = time.perf_counter()
    py_result = sim.run(time_steps=kgrid.Nt, dt=kw_dt)
    py_time = time.perf_counter() - start_time
    py_p = py_result.sensor_data
    
    if kw_p.ndim == 2 and kw_p.shape[1] < kw_p.shape[0]:
        kw_p = kw_p.T
        
    num_sensors = py_p.shape[0]
    n_common_ts = min(py_p.shape[1], kw_p.shape[1])
    
    for i in range(num_sensors):
        kw_trace = kw_p[i, :n_common_ts]
        py_trace = py_p[i, :n_common_ts]
        
        diff = kw_trace - py_trace
        l2_err = np.linalg.norm(diff) / np.linalg.norm(kw_trace)
        
        corr = 1.0
        if np.std(kw_trace) > 1e-10 and np.std(py_trace) > 1e-10:
            corr = np.corrcoef(kw_trace, py_trace)[0, 1]
            
        kw_peak_idx = np.argmax(np.abs(kw_trace))
        py_peak_idx = np.argmax(np.abs(py_trace))
        
        kw_peak_t = kw_peak_idx * kw_dt * 1e6
        py_peak_t = py_peak_idx * kw_dt * 1e6
        
        kw_peak_val = kw_trace[kw_peak_idx]
        py_peak_val = py_trace[py_peak_idx]
        
        ratio = kw_peak_val / py_peak_val if abs(py_peak_val) > 1e-10 else 0.0
        
        print(f"Sensor {i+1}:")
        print(f"  - L2 Error: {l2_err:.4f}")
        print(f"  - Correlation: {corr:.4f}")
        print(f"  - Peak time: kw={kw_peak_t:.2f}µs, py={py_peak_t:.2f}µs, Δ={abs(kw_peak_t - py_peak_t):.2f}µs")
        print(f"  - Peak amp: kw={kw_peak_val:.2e}, py={py_peak_val:.2e}, ratio={ratio:.3f}")
        
    print("=" * 80)
    print("RAW TRACE DUMP FOR SENSOR 2 (FIRST 80 STEPS):")
    for step in range(80):
        t_us = step * kw_dt * 1e6
        kw_val = kw_p[1, step]
        py_val = py_p[1, step]
        print(f" t={t_us:05.2f}µs | kw: {kw_val:+.4e} | py: {py_val:+.4e}")

if __name__ == '__main__':
    main()
