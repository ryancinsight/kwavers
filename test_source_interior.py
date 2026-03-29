#!/usr/bin/env python3
"""Test point source INSIDE free field (not in PML) vs k-Wave."""
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
freq = 0.5e6; amp = 1e5; n_cycles = 3
t_end = 30e-6; nt = int(t_end / dt)

t_arr = np.arange(nt) * dt
n_periods = n_cycles
t_burst = n_periods / freq
env = np.where(t_arr < t_burst, 0.5*(1-np.cos(2*np.pi*freq*t_arr/n_periods)), 0.0)
signal = amp * env * np.sin(2*np.pi*freq*t_arr)
print(f"N={N}, dx={dx:.3e}, dt={dt:.4e}, nt={nt}")
print(f"pml_size={pml_size}, free_field=cells {pml_size}..{N-1-pml_size}")

# Test different source positions
positions = [
    ("in_PML",       4, 8),    # source in PML (x=4)
    ("PML_boundary", 6, 10),   # source at PML boundary
    ("free_field",   10, 14),  # source in free field
    ("center",       12, 16),  # near center
]

for label, sx, sensor_x in positions:
    source_mask = np.zeros((N,N,N)); source_mask[sx, N//2, N//2] = 1.0
    sensor_mask = np.zeros((N,N,N), dtype=bool); sensor_mask[sensor_x, N//2, N//2] = True

    # k-Wave
    kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    ks.p_mask = source_mask.astype(bool)
    ks.p = signal.reshape(1, -1)
    ks.p_mode = "additive-no-correction"
    ksen = kSensor(sensor_mask); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                           simulation_options=so, execution_options=eo)
    p_kw = np.array(r["p"]).flatten().astype(float)
    rms_kw = np.sqrt(np.mean(p_kw**2))
    peak_kw = np.max(np.abs(p_kw))

    # kwavers
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    source = kw.Source.from_mask(source_mask.astype(np.float64), signal,
                                  frequency=freq, mode="additive_no_correction")
    sensor = kw.Sensor.point(position=(sensor_x*dx, (N//2)*dx, (N//2)*dx))
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=nt, dt=dt)
    p_kwa = np.array(result.sensor_data).flatten().astype(float)
    if len(p_kwa) > nt: p_kwa = p_kwa[1:]
    rms_kwa = np.sqrt(np.mean(p_kwa**2))
    peak_kwa = np.max(np.abs(p_kwa))

    n = min(len(p_kw), len(p_kwa))
    corr = np.corrcoef(p_kw[:n], p_kwa[:n])[0,1]
    print(f"{label:15s} src=x{sx} sens=x{sensor_x}: "
          f"corr={corr:.4f}, peak_ratio={peak_kwa/peak_kw:.4f}, rms_ratio={rms_kwa/rms_kw:.4f} "
          f"(kw_peak={peak_kw:.3e}, kwa_peak={peak_kwa:.3e})")
