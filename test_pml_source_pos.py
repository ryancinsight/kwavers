#!/usr/bin/env python3
"""Test plane wave amplitude with source at different z positions (PML vs active domain)."""
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

N = 32; dx = 0.2e-3; c = 1500.0; rho = 1000.0
pml_size = 6; cfl = 0.3; dt = cfl * dx / c
freq = 1e6; amp = 1e5
t_end = 30e-6; nt = int(t_end / dt)

t_arr = np.arange(nt) * dt
signal = amp * np.sin(2*np.pi*freq*t_arr)
print(f"N={N}, dx={dx:.3e}, dt={dt:.4e}, nt={nt}, pml_size={pml_size}")
print(f"PML region: z=[0..{pml_size-1}]  active domain: z=[{pml_size}..{N-pml_size-1}]")

# Sensor: at z = N//2 = 16 (center)
# Source positions to test
source_z_positions = [0, pml_size//2, pml_size, pml_size+2, N//2-4]

def run_kwave(source_z):
    kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    sm = np.zeros((N,N,N)); sm[:,:,source_z] = 1.0
    ks.p_mask = sm.astype(bool)
    n_pts = int(np.sum(sm))
    ks.p = np.tile(signal, (n_pts, 1))
    # Use additive-no-correction to match kwavers additive_no_correction
    ks.p_mode = "additive-no-correction"
    sensor_mask = np.zeros((N,N,N),dtype=bool); sensor_mask[N//2,N//2,N//2] = True
    ksen = kSensor(sensor_mask); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                            simulation_options=so, execution_options=eo)
    return np.array(r["p"]).flatten().astype(float)

def run_kwa(source_z):
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    sm = np.zeros((N,N,N)); sm[:,:,source_z] = 1.0
    source = kw.Source.from_mask(sm.astype(np.float64), signal,
                                  frequency=freq, mode='additive_no_correction')
    sensor = kw.Sensor.point(position=((N//2)*dx, (N//2)*dx, (N//2)*dx))
    sim = kw.Simulation(grid, medium, source, sensor,
                        solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=nt, dt=dt)
    d = np.array(result.sensor_data).flatten().astype(float)
    if len(d) > nt: d = d[1:]
    return d

print(f"\n{'src_z':>6}  {'in_PML?':>8}  {'kw_rms':>12}  {'kwa_rms':>12}  {'ratio':>8}  {'corr':>8}")
print("-"*70)

for src_z in source_z_positions:
    in_pml = "YES" if src_z < pml_size else "no"
    # Check if sensor would be between source and far PML
    sensor_z = N//2
    if src_z >= sensor_z:
        print(f"  {src_z:>4} ({in_pml:>5}): SKIP - source behind sensor")
        continue

    p_kw = run_kwave(src_z)
    p_kwa = run_kwa(src_z)

    n = min(len(p_kw), len(p_kwa))
    w = slice(n//2, n)  # steady-state window
    rms_kw = np.sqrt(np.mean(p_kw[w]**2))
    rms_kwa = np.sqrt(np.mean(p_kwa[w]**2))
    ratio = rms_kwa / (rms_kw + 1e-30)
    corr = np.corrcoef(p_kw[:n], p_kwa[:n])[0,1]

    print(f"  {src_z:>4} ({in_pml:>5}):  {rms_kw:>12.4e}  {rms_kwa:>12.4e}  {ratio:>8.4f}  {corr:>8.4f}")
