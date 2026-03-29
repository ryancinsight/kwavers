#!/usr/bin/env python3
"""Compare z-axis amplitude profile for plane wave between k-Wave and kwavers."""
import numpy as np
import pykwavers as kw
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.data import Vector

N = 32; dx = 1e-3; c = 1500.0; rho = 1000.0
pml_size = 6; cfl = 0.3; dt = cfl * dx / c
freq = 0.5e6; amp = 1e5; nt = 100
t_arr = np.arange(nt) * dt
signal = amp * np.sin(2*np.pi*freq*t_arr)

# Source: plane wave at z=0
sm = np.zeros((N,N,N)); sm[:,:,0] = 1.0

# Sensor: z-axis at (N//2, N//2, all z)
sensor_line = np.zeros((N,N,N), dtype=bool)
sensor_line[N//2, N//2, :] = True

print(f"N={N}, pml_size={pml_size}, nt={nt}")
print(f"PML region: z=[0..{pml_size-1}], active: z=[{pml_size}..{N-pml_size-1}]")
print(f"Source: z=0 face ({N}x{N}={N*N} points)")

# k-Wave
kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
kgrid.setTime(nt, dt)
km = kWaveMedium(sound_speed=c, density=rho)
ks = kSource()
ks.p_mask = sm.astype(bool)
ks.p = signal.reshape(1,-1)
ks.p_mode = "additive-no-correction"
ksen = kSensor(sensor_line); ksen.record = ["p"]
so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                        simulation_options=so, execution_options=eo)
# p shape: (N, nt) - one row per sensor point, time columns
p_kw = np.array(r["p"])
# p_kw shape might be (N, nt) already
if p_kw.ndim == 1:
    p_kw = p_kw.reshape(-1, nt)
print(f"\nk-Wave sensor output shape: {p_kw.shape}")

# kwavers - using from_mask sensor to get z-axis
grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
source = kw.Source.from_mask(sm.astype(np.float64), signal, frequency=freq,
                              mode='additive_no_correction')
sensor = kw.Sensor.from_mask(sensor_line)
sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD, pml_size=pml_size)
result = sim.run(time_steps=nt, dt=dt)
p_kwa = np.array(result.sensor_data)
# kwavers returns (N_sensors, nt+1) or (N_sensors, nt)
if p_kwa.ndim == 1:
    p_kwa = p_kwa.reshape(1, -1)
print(f"kwavers sensor output shape: {p_kwa.shape}")
if p_kwa.shape[1] == nt+1:
    p_kwa = p_kwa[:, 1:]  # trim t=0
print(f"kwavers sensor output (trimmed) shape: {p_kwa.shape}")

# Compute RMS at each z position (last half of simulation = steady state)
w = slice(nt//2, nt)
rms_kw = np.sqrt(np.mean(p_kw[:,w]**2, axis=1))
rms_kwa = np.sqrt(np.mean(p_kwa[:,w]**2, axis=1))

print(f"\n{'z':>4}  {'region':>8}  {'kw_rms':>12}  {'kwa_rms':>12}  {'ratio':>8}")
print("-"*55)
for k in range(N):
    region = "PML" if k < pml_size or k >= N-pml_size else "active"
    ratio = rms_kwa[k] / (rms_kw[k] + 1e-30) if rms_kw[k] > 100 else float('nan')
    print(f"{k:>4}  {region:>8}  {rms_kw[k]:>12.4e}  {rms_kwa[k]:>12.4e}  {ratio:>8.4f}")
