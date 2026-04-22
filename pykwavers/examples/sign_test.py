"""Quick sign-convention diagnostic for pykwavers CW source."""
import numpy as np
import sys
sys.path.insert(0, '.')
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()
import pykwavers as pkw

# Minimal test: 32^3 grid, single source, single sensor
N = 32; dx = 1e-3; c0 = 1500.0; rho0 = 1000.0
dt = 0.3 * dx / c0   # CFL 0.3 → dt = 2e-7 s
Nt = 30  # a few steps

# Source at center [16,16,16], sensor at [24,16,16] (8 pts in +x)
src_mask = np.zeros((N, N, N))
src_mask[16, 16, 16] = 1.0

# Simple signal: smooth ramp starting at 0
t_arr = np.arange(Nt) * dt
freq = c0 / (8 * dx)  # wavelength = 8 dx
signal = np.sin(2 * np.pi * freq * t_arr)
print(f"Signal first 10: {signal[:10].round(4).tolist()}")

source = pkw.Source.from_mask(src_mask, signal, float(freq), mode='additive')

sens_mask = np.zeros((N, N, N), dtype=bool)
sens_mask[24, 16, 16] = True
sensor = pkw.Sensor.from_mask(sens_mask)

grid = pkw.Grid(N, N, N, dx, dx, dx)
medium = pkw.Medium.homogeneous(sound_speed=c0, density=rho0)
sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
sim.set_pml_size(4)
sim.set_pml_inside(True)

result = sim.run(time_steps=Nt, dt=dt)
sd = np.asarray(result.sensor_data)
sd_flat = sd.flatten() if sd.ndim == 1 else sd[0]

print(f"Sensor data shape: {sd.shape}")
print(f"Sensor first {Nt} steps: {sd_flat.round(6).tolist()}")

arrival_step = int(round(8 * dx / c0 / dt))
print(f"\nExpected arrival step: {arrival_step} (8 dx at c0)")
print(f"Signal at arrival: {signal[arrival_step]:.4f}")
print(f"Sensor at arrival: {sd_flat[arrival_step]:.6f}")
print(f"Signal sign=+, Sensor sign={'POSITIVE' if sd_flat[arrival_step] > 0 else 'NEGATIVE'}")
print(f"\nConclusion: source injects positive signal. Sensor should be POSITIVE at arrival.")
print(f"pykwavers sign at sensor: {'CORRECT (+)' if sd_flat[arrival_step] > 0 else 'WRONG (-)'}")
