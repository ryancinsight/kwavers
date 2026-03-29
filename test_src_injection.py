#!/usr/bin/env python3
"""Minimal source injection amplitude test - check raw injection level."""
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
pml_size = 6
dt = 0.3 * dx / c  # CFL=0.3

print(f"N={N}, dx={dx:.3e}, dt={dt:.4e}, c={c}")
print(f"Expected: 1/(3*c^2) = {1/(3*c**2):.4e}")

def run_kwave_single_step(source_mask, signal, nt_run, mode=None):
    """Run k-Wave for nt_run steps and return sensor data."""
    kgrid = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))
    kgrid.setTime(nt_run, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    ks.p_mask = source_mask.astype(bool)
    # Same signal for all mask points
    n_pts = int(np.sum(source_mask))
    ks.p = np.tile(signal[:nt_run], (n_pts, 1))
    if mode is not None:
        ks.p_mode = mode
    # Record entire field at all points in a slice
    sensor_mask = np.ones((N, N, N), dtype=bool)
    ksen = kSensor(sensor_mask); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="double", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                            simulation_options=so, execution_options=eo)
    p = np.array(r["p"])
    # reshape: (N*N*N, nt_run)
    p = p.reshape(N, N, N, nt_run)
    return p

def run_kwa_single_step(source_mask, signal, nt_run, mode='additive'):
    """Run kwavers for nt_run steps with full-field sensor."""
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    source = kw.Source.from_mask(source_mask.astype(np.float64), signal[:nt_run],
                                  frequency=c/dx, mode=mode)
    sensor_mask = np.ones((N, N, N), dtype=bool)
    sensor = kw.Sensor.from_mask(sensor_mask)
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=nt_run, dt=dt)
    p = np.array(result.sensor_data)
    # p shape: (N*N*N, nt_run+1) or (N*N*N, nt_run)
    if p.shape[1] == nt_run + 1:
        p = p[:, 1:]  # skip t=0
    p = p.reshape(N, N, N, nt_run)
    return p

amp = 1e5
nt_run = 2  # just 2 steps to see raw injection

# Step signal: constant at amp for both steps
signal = np.ones(nt_run) * amp

print("\n=== POINT SOURCE (1 point at center) ===")
sx, sy, sz = N//2, N//2, N//2
sm_point = np.zeros((N,N,N)); sm_point[sx, sy, sz] = 1.0

print("Running k-Wave (2 steps, point source)...")
p_kw_pt = run_kwave_single_step(sm_point, signal, nt_run)
print(f"  k-Wave: p at source point after step 1: {p_kw_pt[sx,sy,sz,0]:.4e}")
print(f"  k-Wave: p at source+1 after step 1: {p_kw_pt[sx+1,sy,sz,0]:.4e}")
print(f"  k-Wave: max pressure after step 1: {np.max(np.abs(p_kw_pt[:,:,:,0])):.4e}")

print("Running kwavers (2 steps, point source, additive)...")
p_kwa_pt = run_kwa_single_step(sm_point, signal, nt_run)
print(f"  kwavers: p at source point after step 1: {p_kwa_pt[sx,sy,sz,0]:.4e}")
print(f"  kwavers: p at source+1 after step 1: {p_kwa_pt[sx+1,sy,sz,0]:.4e}")
print(f"  kwavers: max pressure after step 1: {np.max(np.abs(p_kwa_pt[:,:,:,0])):.4e}")

print(f"\n  Point source ratio (kwa/kw): {p_kwa_pt[sx,sy,sz,0]/p_kw_pt[sx,sy,sz,0]:.4f}")

print("\n=== PLANE WAVE SOURCE (full z=0 face, N*N points) ===")
sm_plane = np.zeros((N,N,N)); sm_plane[:,:,0] = 1.0
n_pts_plane = int(np.sum(sm_plane))
print(f"  Number of source points: {n_pts_plane}")

print("Running k-Wave (2 steps, plane wave)...")
p_kw_pl = run_kwave_single_step(sm_plane, signal, nt_run)
print(f"  k-Wave: p at z=0 center after step 1: {p_kw_pl[N//2,N//2,0,0]:.4e}")
print(f"  k-Wave: p at z=1 center after step 1: {p_kw_pl[N//2,N//2,1,0]:.4e}")
print(f"  k-Wave: p at z=2 center after step 1: {p_kw_pl[N//2,N//2,2,0]:.4e}")
print(f"  k-Wave: max pressure after step 1: {np.max(np.abs(p_kw_pl[:,:,:,0])):.4e}")

print("Running kwavers (2 steps, plane wave, additive)...")
p_kwa_pl = run_kwa_single_step(sm_plane, signal, nt_run)
print(f"  kwavers: p at z=0 center after step 1: {p_kwa_pl[N//2,N//2,0,0]:.4e}")
print(f"  kwavers: p at z=1 center after step 1: {p_kwa_pl[N//2,N//2,1,0]:.4e}")
print(f"  kwavers: p at z=2 center after step 1: {p_kwa_pl[N//2,N//2,2,0]:.4e}")
print(f"  kwavers: max pressure after step 1: {np.max(np.abs(p_kwa_pl[:,:,:,0])):.4e}")

print(f"\n  Plane wave ratio (kwa/kw) at z=0: {p_kwa_pl[N//2,N//2,0,0]/(p_kw_pl[N//2,N//2,0,0]+1e-30):.4f}")

print("\n=== RATIO SUMMARY ===")
print(f"  Point source ratio:  {p_kwa_pt[sx,sy,sz,0]/p_kw_pt[sx,sy,sz,0]:.4f}")
print(f"  Plane wave ratio:    {p_kwa_pl[N//2,N//2,0,0]/(p_kw_pl[N//2,N//2,0,0]+1e-30):.4f}")
