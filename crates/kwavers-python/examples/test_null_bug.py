import numpy as np
import pykwavers as kw

def main():
    Nx = 88; Ny = 108; Nz = 44
    grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=1e-4, dy=1e-4, dz=1e-4)
    medium = kw.Medium.homogeneous(sound_speed=1540.0, density=1000.0)
    
    source_mask = np.zeros((Nx, Ny, Nz))
    source_mask[10, 50, 20] = 1.0
    
    # Non-zero signal
    signal_to_use = np.sin(np.linspace(0, 10, 100)) * 1e6
    source = kw.Source.from_mask(source_mask, signal_to_use, frequency=0.5e6)
    
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_mask[12, 50, 20] = True
    sensor = kw.Sensor.from_mask(sensor_mask)
    
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.FDTD)
    result = sim.run(time_steps=100, dt=1e-7)
    p = result.sensor_data
    
    print("="*40)
    print("MINIMAL TEST RESULT")
    print("="*40)
    print(f'Max: {np.max(p)}')
    print(f'Min: {np.min(p)}')
    print(f'NaNs: {np.isnan(p).sum()}')
    print("="*40)

if __name__ == '__main__':
    main()
