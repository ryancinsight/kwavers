#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))
import pykwavers as kw

grid_size = 64
spacing = 0.1e-3
c0 = 1500.0
rho0 = 1000.0
freq = 1e6
amplitude = 1e5
duration = 5e-6  # Longer
cfl = 0.3

dx = spacing
dt = cfl * dx / c0
nt = int(duration / dt)

print(f"Grid: {grid_size}³, dt={dt*1e9:.2f} ns, steps={nt}")

grid = kw.Grid(nx=grid_size, ny=grid_size, nz=grid_size, dx=spacing, dy=spacing, dz=spacing)
medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
mask[:, :, 0] = 1.0

t = np.arange(nt) * dt
signal = amplitude * np.sin(2 * np.pi * freq * t)

source = kw.Source.from_mask(mask, signal, frequency=freq)
sensor = kw.Sensor.point(position=(grid_size//2 * spacing, grid_size//2 * spacing, grid_size//2 * spacing))

print("\nRunning PSTD...")
sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
result = sim.run(time_steps=nt, dt=dt)

max_amp = np.abs(result.sensor_data).max()
print(f"\nResult:")
print(f"  Max amplitude: {max_amp:.3e} Pa ({max_amp*1e-3:.1f} kPa)")
print(f"  Expected:      {amplitude:.3e} Pa ({amplitude*1e-3:.1f} kPa)")
print(f"  Ratio: {max_amp / amplitude:.2f}x")
