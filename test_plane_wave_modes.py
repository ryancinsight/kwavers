#!/usr/bin/env python3
"""Test plane wave amplitude with different source modes in k-Wave and kwavers."""
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
freq = 0.5e6; amp = 1e5
t_end = 40e-6; nt = int(t_end / dt)

t_arr = np.arange(nt) * dt
signal = amp * np.sin(2*np.pi*freq*t_arr)
print(f"N={N}, dx={dx:.3e}, dt={dt:.4e}, nt={nt}")
print(f"lambda/dx={c/freq/dx:.1f}, CFL={cfl}")

# Plane wave: full z=0 face
sm_plane = np.zeros((N,N,N)); sm_plane[:,:,0] = 1.0
n_pts = int(np.sum(sm_plane))
# Sensor: center of domain
sx, sy, sz = N//2, N//2, N//2
sensor_mask = np.zeros((N,N,N),dtype=bool); sensor_mask[sx,sy,sz] = True

def run_kwave(source_mask, sig, mode):
    kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    ks.p_mask = source_mask.astype(bool)
    ks.p = np.tile(sig, (int(np.sum(source_mask)), 1))
    ks.p_mode = mode
    ksen = kSensor(sensor_mask); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                            simulation_options=so, execution_options=eo)
    return np.array(r["p"]).flatten().astype(float)

def run_kwa(source_mask, sig, mode):
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    source = kw.Source.from_mask(source_mask.astype(np.float64), sig,
                                  frequency=freq, mode=mode)
    sensor = kw.Sensor.point(position=(sx*dx, sy*dx, sz*dx))
    sim = kw.Simulation(grid, medium, source, sensor,
                        solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=nt, dt=dt)
    d = np.array(result.sensor_data).flatten().astype(float)
    if len(d) > nt: d = d[1:]
    return d

def stats(label, p_kw, p_kwa):
    n = min(len(p_kw), len(p_kwa))
    a, b = p_kw[:n], p_kwa[:n]
    # Steady-state window (after 2 wavelengths have passed, last 50%)
    w = slice(n//2, n)
    rms_kw = np.sqrt(np.mean(a[w]**2))
    rms_kwa = np.sqrt(np.mean(b[w]**2))
    rms_ratio = rms_kwa / (rms_kw + 1e-30)
    corr = np.corrcoef(a, b)[0,1]
    print(f"  {label}: corr={corr:.4f}, rms_ratio={rms_ratio:.4f} "
          f"(kw={rms_kw:.3e}, kwa={rms_kwa:.3e})")

print("\n=== PLANE WAVE ===")
for kw_mode, kwa_mode in [
    ("additive-no-correction", "additive_no_correction"),
    ("additive", "additive"),
]:
    print(f"\n  k-Wave mode: {kw_mode}, kwavers mode: {kwa_mode}")
    p_kw = run_kwave(sm_plane, signal, kw_mode)
    p_kwa = run_kwa(sm_plane, signal, kwa_mode)
    stats(f"kw:{kw_mode}", p_kw, p_kwa)

# Now test POINT source for comparison
sm_pt = np.zeros((N,N,N)); sm_pt[4, N//2, N//2] = 1.0
sens_pt = np.zeros((N,N,N),dtype=bool); sens_pt[8, N//2, N//2] = True

def run_kwave_pt(sig, mode):
    kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    ks.p_mask = sm_pt.astype(bool)
    ks.p = sig.reshape(1,-1)
    ks.p_mode = mode
    ksen = kSensor(sens_pt); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                            simulation_options=so, execution_options=eo)
    return np.array(r["p"]).flatten().astype(float)

def run_kwa_pt(sig, mode):
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    source = kw.Source.from_mask(sm_pt.astype(np.float64), sig, frequency=freq, mode=mode)
    sensor = kw.Sensor.point(position=(8*dx, (N//2)*dx, (N//2)*dx))
    sim = kw.Simulation(grid, medium, source, sensor,
                        solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=len(sig), dt=dt)
    d = np.array(result.sensor_data).flatten().astype(float)
    if len(d) > nt: d = d[1:]
    return d

# Use nt_short = 100 for point source (just the burst region)
nt_short = 100
sig_short = signal[:nt_short]

print("\n=== POINT SOURCE (for comparison) ===")
for kw_mode, kwa_mode in [
    ("additive-no-correction", "additive_no_correction"),
    ("additive", "additive"),
]:
    print(f"\n  k-Wave mode: {kw_mode}, kwavers mode: {kwa_mode}")
    p_kw = run_kwave_pt(sig_short, kw_mode)
    p_kwa = run_kwa_pt(sig_short, kwa_mode)
    n = min(len(p_kw), len(p_kwa))
    a, b = p_kw[:n], p_kwa[:n]
    corr = np.corrcoef(a, b)[0,1]
    rms_kw = np.sqrt(np.mean(a**2))
    rms_kwa = np.sqrt(np.mean(b**2))
    print(f"  corr={corr:.4f}, rms_ratio={rms_kwa/rms_kw:.4f} (kw={rms_kw:.3e}, kwa={rms_kwa:.3e})")
