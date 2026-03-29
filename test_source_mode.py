#!/usr/bin/env python3
"""Test additive vs additive_no_correction source modes vs k-Wave."""
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

# Simulation params
N = 32; dx = 1e-3; c = 1500.0; rho = 1000.0
cfl = 0.3; dt = cfl * dx / c
pml_size = 6
freq = 0.5e6; amp = 1e5; n_cycles = 3
t_end = 20e-6; nt = int(t_end / dt)

print(f"N={N}, dx={dx:.3e}, dt={dt:.4e}, nt={nt}, freq={freq:.2e}")
print(f"lambda/dx = {c/freq/dx:.1f}  CFL = {cfl}")

# Source: single point at (4, N//2, N//2)
sx, sy, sz = 4, N//2, N//2
source_mask = np.zeros((N, N, N)); source_mask[sx, sy, sz] = 1.0

# Sensor: 4 cells away from source in x
sens_x = sx + 4
sensor_mask = np.zeros((N, N, N), dtype=bool); sensor_mask[sens_x, sy, sz] = True

# Tone burst signal (n_cycles periods)
t_arr = np.arange(nt) * dt
n_periods = n_cycles
t_burst = n_periods / freq
env = np.where(t_arr < t_burst, 0.5 * (1 - np.cos(2*np.pi*freq*t_arr / n_periods)), 0.0)
signal = amp * env * np.sin(2*np.pi*freq*t_arr)

print(f"\nSignal: peak={np.max(np.abs(signal)):.4e}, burst duration={t_burst*1e6:.1f} us")

# ---- k-Wave ----
print("\n=== Running k-Wave ===")
kgrid = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))
kgrid.setTime(nt, dt)
km = kWaveMedium(sound_speed=c, density=rho)
ks = kSource()
ks.p_mask = source_mask.astype(bool)
ks.p = signal.reshape(1, -1)
ksen = kSensor(sensor_mask); ksen.record = ["p"]
so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
rkw = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                          simulation_options=so, execution_options=eo)
p_kw = np.array(rkw["p"]).flatten().astype(float)
print(f"k-Wave: shape={p_kw.shape}, max={np.max(np.abs(p_kw)):.4e}")

def run_kwa(mode):
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    source = kw.Source.from_mask(source_mask.astype(np.float64), signal, frequency=freq,
                                  mode=mode)
    sensor = kw.Sensor.point(position=(sens_x*dx, sy*dx, sz*dx))
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=nt, dt=dt)
    data = np.array(result.sensor_data).flatten().astype(float)
    # kwavers records t=0 + Nt steps = Nt+1 samples; trim to Nt
    if len(data) > nt:
        data = data[1:]
    return data

def compare(p_kw, p_kwa, label):
    n = min(len(p_kw), len(p_kwa))
    a, b = p_kw[:n], p_kwa[:n]
    corr = np.corrcoef(a, b)[0, 1]
    rms_kw = np.sqrt(np.mean(a**2))
    rms_kwa = np.sqrt(np.mean(b**2))
    rms_ratio = rms_kwa / (rms_kw + 1e-30)
    peak_kw = np.max(np.abs(a))
    peak_kwa = np.max(np.abs(b))
    amp_ratio = peak_kwa / (peak_kw + 1e-30)
    print(f"\n{label}:")
    print(f"  corr={corr:.4f}, rms_ratio={rms_ratio:.4f}, amp_ratio={amp_ratio:.4f}")
    print(f"  k-Wave peak={peak_kw:.4e}, kwavers peak={peak_kwa:.4e}")

    # Step-by-step comparison around main pulse
    peak_idx = np.argmax(np.abs(b))
    print(f"  Main pulse at step {peak_idx}:")
    for i in range(max(0, peak_idx-5), min(n, peak_idx+10)):
        print(f"    step {i:4d}: kw={a[i]:+.4e}  kwa={b[i]:+.4e}  ratio={b[i]/(a[i]+1e-30):+.3f}")
    return corr, rms_ratio, amp_ratio

print("\n=== Running kwavers additive ===")
p_add = run_kwa('additive')
print(f"kwavers additive: shape={p_add.shape}, max={np.max(np.abs(p_add)):.4e}")
corr_add, rms_add, amp_add = compare(p_kw, p_add, "additive")

print("\n=== Running kwavers additive_no_correction ===")
p_nocc = run_kwa('additive_no_correction')
print(f"kwavers no-correction: shape={p_nocc.shape}, max={np.max(np.abs(p_nocc)):.4e}")
corr_nocc, rms_nocc, amp_nocc = compare(p_kw, p_nocc, "additive_no_correction")

print("\n=== Summary ===")
print(f"{'mode':<30} {'corr':>8} {'rms_ratio':>10} {'amp_ratio':>10}")
print(f"{'additive':<30} {corr_add:>8.4f} {rms_add:>10.4f} {amp_add:>10.4f}")
print(f"{'additive_no_correction':<30} {corr_nocc:>8.4f} {rms_nocc:>10.4f} {amp_nocc:>10.4f}")
