import numpy as np
import pykwavers as kwa
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.data import Vector

def run_comparison():
    # 1. Grid
    nx, ny, nz = 64, 64, 64
    dx = 0.1e-3
    kgrid = kWaveGrid(Vector([nx, ny, nz]), Vector([dx, dx, dx]))
    kgrid.makeTime(1500, cfl=0.3, t_end=20 * dx / 1500)
    kwa_grid = kwa.Grid(nx, ny, nz, dx, dx, dx)

    # 2. Medium
    medium = kWaveMedium(sound_speed=1500.0, density=1000.0)
    kwa_medium = kwa.Medium.homogeneous(1500.0, 1000.0)

    # 3. Source (Initial Pressure p0)
    xx, yy, zz = np.meshgrid(np.arange(nx) - 32, np.arange(ny) - 32, np.arange(nz) - 32, indexing='ij')
    r2 = xx**2 + yy**2 + zz**2
    p0 = 2.0 * np.exp(-r2 / (2 * 3**2))  # sigma = 3 grid points
    
    
    source = kSource()
    source.p0 = p0
    kwa_source = kwa.Source.from_initial_pressure(p0)

    # 4. Sensor - place strictly inside the 16^3 unattenuated core
    sensor_mask = np.zeros((nx, ny, nz), dtype=bool)
    sensor_mask[32, 32, 32] = True
    sensor_mask[32, 38, 32] = True

    sensor = kSensor(mask=sensor_mask)
    kwa_sensor = kwa.Sensor.from_mask(sensor_mask)

    sim_options = SimulationOptions(
        pml_inside=True, 
        pml_size=10, 
        data_cast='double', 
        save_to_disk=True,
        smooth_p0=False,
    )
    exec_options = SimulationExecutionOptions()

    print(f"Running k-wave-python...")
    kw_result = kspaceFirstOrder3D(
        kgrid=kgrid, medium=medium, source=source, sensor=sensor,
        simulation_options=sim_options,
        execution_options=exec_options
    )
    
    # kwave returns a dictionary if multiple metrics, or direct array if just p
    if isinstance(kw_result, dict):
        kw_sensor_data = kw_result['p']
    elif hasattr(kw_result, "p"):
        kw_sensor_data = kw_result.p
    else:
        kw_sensor_data = kw_result

    print(f"Running pykwavers...")
    time_steps = kgrid.Nt
    kwa_sim = kwa.Simulation(
        kwa_grid, 
        kwa_medium, 
        kwa_source, 
        kwa_sensor, 
        solver=kwa.SolverType.PSTD,
        pml_size=10
    )
    kwa_result = kwa_sim.run(time_steps=time_steps, dt=kgrid.dt)

    kwa_sensor_data_raw = kwa_result.sensor_data

    # 5. Compare
    print(f"kw_sensor_data shape: {np.shape(kw_sensor_data)}")
    print(f"kwa_sensor_data shape: {np.shape(kwa_sensor_data_raw)}")
    
    if kw_sensor_data.ndim == 1:
        kw_sensor_data = kw_sensor_data.reshape(-1, 1)
    else:
        kw_sensor_data = kw_sensor_data.T
    kwa_sensor_data = kwa_sensor_data_raw

    print(f"kw_sensor_data shape: {kw_sensor_data.shape}")
    print(f"kwa_sensor_data shape: {kwa_sensor_data.shape}")

    # Align temporal samples:
    # k-wave records t=0 before loop.
    # kwavers records after t=1 step.
    # We compare indices 1..N of kw_data with 0..N-1 of kwa_data.
    kw_aligned = kw_sensor_data[:, 1:]
    kwa_aligned = kwa_sensor_data[:, :-1]

    diff = np.abs(kw_aligned.astype(float) - kwa_aligned.astype(float))
    max_diff = np.max(diff)

    print(f"k-wave max: {np.max(kw_aligned):.4e}, min: {np.min(kw_aligned):.4e}")
    print(f"kwavers max: {np.max(kwa_aligned):.4e}, min: {np.min(kwa_aligned):.4e}")
    print(f"Max absolute diff: {max_diff:.4e}")
    
    if max_diff < 1e-4:
        print("SIMULATION PASS")
    else:
        print("SIMULATION FAIL")

if __name__ == '__main__':
    run_comparison()
