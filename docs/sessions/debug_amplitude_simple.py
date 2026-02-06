#!/usr/bin/env python3
"""Simplified amplitude diagnostic - pykwavers only"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))
import pykwavers as kw

# Configuration
grid_size = 64
spacing = 0.1e-3  # 0.1 mm
c0 = 1500.0
rho0 = 1000.0
freq = 1e6
amplitude = 1e5  # 100 kPa
duration = 5e-6
cfl = 0.3

dx = spacing
dt = cfl * dx / c0
nt = int(duration / dt)

print("=" * 80)
print(f"Grid: {grid_size}³, dt={dt*1e9:.2f} ns, steps={nt}")
print(f"Source: {freq*1e-6:.1f} MHz, {amplitude*1e-3:.0f} kPa")
print("=" * 80)
print()

# Grid and medium
grid = kw.Grid(nx=grid_size, ny=grid_size, nz=grid_size, dx=spacing, dy=spacing, dz=spacing)
medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

# Source: plane wave at z=0
mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
mask[:, :, 0] = 1.0

t = np.arange(nt) * dt
signal = amplitude * np.sin(2 * np.pi * freq * t)

print(f"Input signal: min={signal.min():.2e}, max={signal.max():.2e}")
print(f"Mask: {np.sum(mask > 0)} active points, value={mask[0,0,0]}")
print()

source = kw.Source.from_mask(mask, signal, frequency=freq)

# Sensor at center
sensor_pos = (grid_size // 2 * spacing, grid_size // 2 * spacing, grid_size // 2 * spacing)
sensor = kw.Sensor.point(position=sensor_pos)

# Test FDTD
print("Running FDTD...")
sim_fdtd = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.FDTD)
result_fdtd = sim_fdtd.run(time_steps=nt, dt=dt)
fdtd_max = np.abs(result_fdtd.sensor_data).max()
print(f"  Output: min={result_fdtd.sensor_data.min():.2e}, max={result_fdtd.sensor_data.max():.2e}")
print(f"  Max amplitude: {fdtd_max:.2e} Pa ({fdtd_max*1e-3:.1f} kPa)")
print(f"  Ratio vs input: {fdtd_max/amplitude:.2f}x")
print()

# Test PSTD
print("Running PSTD...")
sim_pstd = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
result_pstd = sim_pstd.run(time_steps=nt, dt=dt)
pstd_max = np.abs(result_pstd.sensor_data).max()
print(f"  Output: min={result_pstd.sensor_data.min():.2e}, max={result_pstd.sensor_data.max():.2e}")
print(f"  Max amplitude: {pstd_max:.2e} Pa ({pstd_max*1e-3:.1f} kPa)")
print(f"  Ratio vs input: {pstd_max/amplitude:.2f}x")
print()

print("=" * 80)
print("Summary:")
print(f"  Expected:  {amplitude:.2e} Pa ({amplitude*1e-3:.0f} kPa)")
print(f"  FDTD:      {fdtd_max:.2e} Pa ({fdtd_max*1e-3:.1f} kPa) [{fdtd_max/amplitude:.2f}x]")
print(f"  PSTD:      {pstd_max:.2e} Pa ({pstd_max*1e-3:.1f} kPa) [{pstd_max/amplitude:.2f}x]")
print("=" * 80)
