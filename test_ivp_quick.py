#!/usr/bin/env python3
"""Quick IVP parity test."""
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

N = 32; dx = 1e-3; c = 1500.0; rho = 1000.0; pml_size = 6
cfl = 0.3; dt = cfl * dx / c; nt = 60

# Gaussian initial pressure at center
x = np.linspace(0, (N-1)*dx, N)
y = np.linspace(0, (N-1)*dx, N)
z = np.linspace(0, (N-1)*dx, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
cx = cy = cz = (N-1)*dx/2
sig = 3*dx
p0 = 1e4 * np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) / (2*sig**2))

# k-Wave
kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
kgrid.setTime(nt, dt)
km = kWaveMedium(sound_speed=c, density=rho)
ks = kSource(); ks.p0 = p0
sensor_mask = np.zeros((N,N,N), dtype=bool); sensor_mask[N//2+2, N//2, N//2] = True
ksen = kSensor(sensor_mask); ksen.record = ["p"]
so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                       simulation_options=so, execution_options=eo)
p_kw = np.array(r["p"]).flatten().astype(float)

# kwavers
grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
source = kw.Source.from_initial_pressure(p0)
sens_x = (N//2+2)*dx; sens_y = (N//2)*dx; sens_z = (N//2)*dx
sensor = kw.Sensor.point(position=(sens_x, sens_y, sens_z))
sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD, pml_size=pml_size)
result = sim.run(time_steps=nt, dt=dt)
p_kwa = np.array(result.sensor_data).flatten().astype(float)
if len(p_kwa) > nt: p_kwa = p_kwa[1:]

n = min(len(p_kw), len(p_kwa))
corr = np.corrcoef(p_kw[:n], p_kwa[:n])[0,1]
peak_kw = np.max(np.abs(p_kw[:n]))
peak_kwa = np.max(np.abs(p_kwa[:n]))
rms_ratio = np.sqrt(np.mean(p_kwa[:n]**2)) / (np.sqrt(np.mean(p_kw[:n]**2)) + 1e-30)
print(f"IVP parity: corr={corr:.6f}, peak_ratio={peak_kwa/peak_kw:.6f}, rms_ratio={rms_ratio:.6f}")
print(f"  k-Wave peak={peak_kw:.4e}, kwavers peak={peak_kwa:.4e}")
