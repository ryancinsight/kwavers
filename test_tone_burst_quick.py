#!/usr/bin/env python3
"""
Quick tone burst comparison - focuses on timing alignment and phase.
Uses small grid so we can track exactly when the wave arrives.
"""
import numpy as np
import sys
sys.path.insert(0, 'pykwavers/python')
import pykwavers as kw

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    HAS_KWAVE = True
except ImportError:
    HAS_KWAVE = False
    print("k-wave-python not found, skipping comparison")
    sys.exit(0)

# Small grid: 32^3, 5 cycles of 500kHz tone burst
N = 32
dx = 2e-3  # 2mm
c0 = 1500.0
rho0 = 1000.0
f0 = 500e3  # 500 kHz
pml_size = 4
src_ix = N // 2  # 16

kgrid_tmp = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid_tmp.makeTime(c0)
dt = float(kgrid_tmp.dt)
print(f"N={N}, dx={dx}, c0={c0}, f0={f0}, dt={dt:.4e}")

# Tone burst: 5 cycles
n_cycles = 5
T = 1.0 / f0
Nt_burst = int(n_cycles * T / dt) + 1
print(f"Tone burst: {n_cycles} cycles, {Nt_burst} steps")

# Total simulation: 2x the burst length
Nt = Nt_burst * 2
t_arr = np.arange(Nt) * dt
signal = np.zeros(Nt)
burst_mask = t_arr < n_cycles * T
signal[burst_mask] = np.sin(2 * np.pi * f0 * t_arr[burst_mask])

# k-Wave
print("\n--- k-Wave ---")
kgrid = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid.setTime(Nt, dt)

src_mask = np.zeros((N, N, N))
src_mask[src_ix, src_ix, src_ix] = 1
kw_source = kSource()
kw_source.p_mask = src_mask
kw_source.p = signal.reshape(1, -1)
kw_source.p_mode = "additive"

# Sensor along x-axis
sen_line = np.zeros((N, N, N), dtype=float)
for x in range(N):
    sen_line[x, src_ix, src_ix] = 1
kw_sensor = kSensor(sen_line)
kw_sensor.record = ["p"]

kw_res = kspaceFirstOrder3D(
    medium=kWaveMedium(sound_speed=c0, density=rho0),
    kgrid=kgrid,
    source=kw_source,
    sensor=kw_sensor,
    simulation_options=SimulationOptions(
        pml_inside=True, pml_size=pml_size, data_cast="double", save_to_disk=True,
    ),
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False, delete_data=True, verbose_level=0),
)
kw_p = np.array(kw_res["p"])
# k-Wave returns (Nt, n_sensors) — transpose to (n_sensors, Nt)
if kw_p.shape[0] == Nt and kw_p.shape[1] != Nt:
    kw_p = kw_p.T
elif kw_p.ndim == 2 and kw_p.shape[0] > kw_p.shape[1]:
    kw_p = kw_p.T
print(f"k-Wave shape: {kw_p.shape}")

# kwavers
print("\n--- kwavers ---")
kwa_grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

py_src_mask = np.zeros((N, N, N), dtype=np.float64)
py_src_mask[src_ix, src_ix, src_ix] = 1.0
kwa_source = kw.Source.from_mask(py_src_mask, signal.copy(), f0, mode="additive")

py_sen_mask = np.zeros((N, N, N), dtype=bool)
for x in range(N):
    py_sen_mask[x, src_ix, src_ix] = True
kwa_sensor = kw.Sensor.from_mask(py_sen_mask)

kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source, kwa_sensor, solver=kw.SolverType.PSTD)
kwa_sim.set_pml_size(pml_size)
kwa_sim.set_pml_inside(True)
kwa_res = kwa_sim.run(time_steps=Nt, dt=dt)
kwa_p_raw = np.array(kwa_res.sensor_data)  # (n_sensors, Nt+1)
print(f"kwavers shape: {kwa_p_raw.shape}")

# Align: kwavers has t=0 initial + Nt steps, k-Wave has Nt steps
# Use kwavers[1:] vs k-Wave[:] for same timing
kwa_p_aligned = kwa_p_raw[:, 1:]  # drop t=0

# Sensor ordering: kwavers Fortran order gives sensors in x-fastest order
# For single-axis x sensor at (x, src_ix, src_ix), sensor index = x (since all unique x)
# After reshape: both have shape (N, Nt)

print(f"\n=== COMPARISON: Source point x={src_ix} ===")
kw_src = kw_p[src_ix, :]
kwa_src = kwa_p_aligned[src_ix, :]
corr = np.corrcoef(kw_src, kwa_src)[0, 1]
rms_ratio = np.sqrt(np.mean(kwa_src**2)) / max(np.sqrt(np.mean(kw_src**2)), 1e-30)
print(f"Source ({src_ix},{src_ix},{src_ix}): corr={corr:.4f}, rms_ratio={rms_ratio:.4f}")

for dist in [2, 4, 6, 8]:
    x = src_ix + dist
    if x >= N:
        continue
    kw_off = kw_p[x, :]
    kwa_off = kwa_p_aligned[x, :]
    corr = np.corrcoef(kw_off, kwa_off)[0, 1] if kw_off.any() or kwa_off.any() else 0.0
    kw_rms = np.sqrt(np.mean(kw_off**2))
    kwa_rms = np.sqrt(np.mean(kwa_off**2))
    rms_r = kwa_rms / max(kw_rms, 1e-30)
    print(f"  off+{dist} ({x},{src_ix},{src_ix}): corr={corr:.4f}, rms_ratio={rms_r:.4f}, kw_rms={kw_rms:.4e}, kwa_rms={kwa_rms:.4e}")

print("\n=== First wave arrival: k-Wave vs kwavers at x=src_ix+4 ===")
x = src_ix + 4
t_arrival = int(4 * dx / c0 / dt)
print(f"Expected arrival step: {t_arrival}")
print(f"{'step':>6} {'k-Wave':>12} {'kwavers':>12} {'ratio':>8}")
for t in range(max(0, t_arrival-3), min(Nt, t_arrival+15)):
    v_kw = kw_p[x, t]
    v_kwa = kwa_p_aligned[x, t]
    ratio = v_kwa/v_kw if abs(v_kw) > 1e-15 else float('nan')
    print(f"{t:6d} {v_kw:12.4e} {v_kwa:12.4e} {ratio:8.3f}")
