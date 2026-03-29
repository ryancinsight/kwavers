#!/usr/bin/env python3
"""Check if k-Wave plane wave amplitude depends on signal format (1 row vs N rows)."""
import numpy as np
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
nt = 200
t_arr = np.arange(nt) * dt
signal = amp * np.sin(2*np.pi*freq*t_arr)
sensor_mask = np.zeros((N,N,N),dtype=bool); sensor_mask[N//2,N//2,N//2] = True

def run_kwave_plane(p_signal, mode="additive-no-correction"):
    kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    sm = np.zeros((N,N,N)); sm[:,:,0] = 1.0
    ks.p_mask = sm.astype(bool)
    ks.p = p_signal
    ks.p_mode = mode
    ksen = kSensor(sensor_mask); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                            simulation_options=so, execution_options=eo)
    p = np.array(r["p"]).flatten().astype(float)
    rms = np.sqrt(np.mean(p[nt//2:]**2))
    return rms, p

n_pts = N*N  # 1024 points

print("Testing k-Wave plane wave with different source signal formats:")
print(f"N={N}, dx={dx:.3e}, dt={dt:.4e}, n_pts={n_pts}")
print()

# 1 row (scalar mode in k-Wave)
sig_1row = signal.reshape(1, -1)
rms_1row, _ = run_kwave_plane(sig_1row, "additive-no-correction")
print(f"1 row  (shape {sig_1row.shape}): RMS = {rms_1row:.4e}")

# N rows (many mode in k-Wave)
sig_nrows = np.tile(signal, (n_pts, 1))
rms_nrows, _ = run_kwave_plane(sig_nrows, "additive-no-correction")
print(f"N rows (shape {sig_nrows.shape}): RMS = {rms_nrows:.4e}")
print(f"Ratio (N/1) = {rms_nrows/rms_1row:.4f}")

# Point source for comparison
def run_kwave_point(p_signal, mode="additive-no-correction"):
    kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    sm_pt = np.zeros((N,N,N)); sm_pt[4,N//2,N//2] = 1.0
    ks.p_mask = sm_pt.astype(bool)
    ks.p = p_signal
    ks.p_mode = mode
    sens_pt = np.zeros((N,N,N),dtype=bool); sens_pt[8,N//2,N//2] = True
    ksen = kSensor(sens_pt); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                            simulation_options=so, execution_options=eo)
    return np.sqrt(np.mean(np.array(r["p"]).flatten()**2))

print()
print("Point source comparison:")
rms_pt = run_kwave_point(signal.reshape(1,-1), "additive-no-correction")
print(f"k-Wave point source RMS: {rms_pt:.4e}")
print(f"k-Wave plane wave (1 row) RMS: {rms_1row:.4e}")
print(f"Plane/Point ratio: {rms_1row/rms_pt:.4f}")
