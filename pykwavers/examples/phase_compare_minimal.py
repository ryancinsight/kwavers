"""
Minimal phase comparison: single point source + single sensor.
k-wave-python vs pykwavers PSTD.
Tests whether the ~8-step phase offset is in propagation or sensor summation.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()
import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import filter_time_series

# --- Same grid as SD compare ---
N = 64; DX = 100e-3 / N; C0 = 1500.0; RHO0 = 1000.0
SOURCE_FREQ = 0.25e6; PML_SIZE = 10
SPHERE_OFFSET = 10; BOWL_RADIUS = N // 2
SRC_IX = int(SPHERE_OFFSET + BOWL_RADIUS)  # 42
SRC_IY = N // 2 - 1   # 31
SRC_IZ = N // 2 - 1   # 31
SENS_IX = SPHERE_OFFSET  # 10 (bowl apex)
SENS_IY = N // 2 - 1    # 31
SENS_IZ = N // 2 - 1    # 31

GRID_SIZE = Vector([N, N, N])
DX_VEC = Vector([DX, DX, DX])
kgrid = kWaveGrid(GRID_SIZE, DX_VEC)
medium_kw = kWaveMedium(sound_speed=C0)
kgrid.makeTime(medium_kw.sound_speed)
dt = float(kgrid.dt); Nt = int(kgrid.Nt)
print(f"Grid {N}^3, dx={DX*1e3:.4f}mm, dt={dt:.3e}s, Nt={Nt}")
print(f"Source [0-idx]: [{SRC_IX},{SRC_IY},{SRC_IZ}], Sensor [0-idx]: [{SENS_IX},{SENS_IY},{SENS_IZ}]")
travel_steps = (SRC_IX - SENS_IX) * DX / C0 / dt
print(f"Travel: {SRC_IX-SENS_IX} dx, expected arrival step = {travel_steps:.1f}")

raw_signal = np.sin(2*np.pi*SOURCE_FREQ*kgrid.t_array)
filtered = filter_time_series(kgrid, medium_kw, raw_signal)
signal_1d = np.asarray(filtered, dtype=np.float64).flatten()

# --- k-Wave ---
src_mask = np.zeros((N,N,N), dtype=np.float64)
src_mask[SRC_IX, SRC_IY, SRC_IZ] = 1.0
source_kw = kSource()
source_kw.p_mask = src_mask
source_kw.p = signal_1d.reshape(1,-1)

sens_mask = np.zeros((N,N,N), dtype=np.int32)
sens_mask[SENS_IX, SENS_IY, SENS_IZ] = 1
sensor_kw = kSensor(sens_mask)

sim_opts = SimulationOptions(pml_size=PML_SIZE, data_cast="single", save_to_disk=True)
exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

print("\n[k-wave] Running...")
import time
t0 = time.perf_counter()
sd_kw = kspaceFirstOrder3D(medium=medium_kw, kgrid=kgrid, source=source_kw, sensor=sensor_kw,
                            simulation_options=sim_opts, execution_options=exec_opts)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
kw_trace = np.asarray(sd_kw["p"], dtype=np.float64).flatten()
print(f"  Shape: {sd_kw['p'].shape}, flattened: {kw_trace.shape}")

# --- pykwavers ---
grid = pkw.Grid(N, N, N, DX, DX, DX)
med = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
source_pkw = pkw.Source.from_mask(src_mask, signal_1d, SOURCE_FREQ, mode="additive")

sens_mask_bool = np.zeros((N,N,N), dtype=bool)
sens_mask_bool[SENS_IX, SENS_IY, SENS_IZ] = True
sensor_pkw = pkw.Sensor.from_mask(sens_mask_bool)

sim = pkw.Simulation(grid, med, source_pkw, sensor_pkw, solver=pkw.SolverType.PSTD)
sim.set_pml_size(PML_SIZE)
sim.set_pml_inside(True)

print("\n[pykwavers] Running...")
t0 = time.perf_counter()
result = sim.run(time_steps=Nt, dt=dt)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
pkw_trace = np.asarray(result.sensor_data, dtype=np.float64).flatten()
print(f"  Shape: {np.asarray(result.sensor_data).shape}")

# --- Comparison ---
arrival = int(round(travel_steps))
win = slice(arrival-5, arrival+15)
print(f"\n[Arrival window step {arrival-5}:{arrival+15}]")
print(f"  kwave : {kw_trace[win].round(6).tolist()}")
print(f"  pkwav : {pkw_trace[win].round(6).tolist()}")

# Find first non-trivial step
kw_max = np.abs(kw_trace).max()
pkw_max = np.abs(pkw_trace).max()
kw_first = next((i for i in range(Nt) if abs(kw_trace[i]) > 0.01*kw_max), -1)
pkw_first = next((i for i in range(Nt) if abs(pkw_trace[i]) > 0.01*pkw_max), -1)
print(f"\n  kwave first_nz idx={kw_first} val={kw_trace[kw_first]:.6f}")
print(f"  pkwav first_nz idx={pkw_first} val={pkw_trace[pkw_first]:.6f}")
print(f"  Arrival step difference: {pkw_first - kw_first} steps")

# Cross-correlation phase
from scipy.signal import correlate
cc = correlate(kw_trace, pkw_trace, mode='full')
lags = np.arange(-(Nt-1), Nt)
peak_lag = lags[np.argmax(cc)]
peak_val = cc.max()
print(f"\n  Cross-corr peak: lag={peak_lag} steps, val={peak_val:.2e}")
print(f"  (positive lag = pykwavers is AHEAD of k-wave)")

# Pearson r
r = np.corrcoef(kw_trace, pkw_trace)[0,1]
print(f"  Pearson r = {r:.6f}")
print(f"  Peak: kwave={kw_max:.6f} Pa, pkwav={pkw_max:.6f} Pa, ratio={pkw_max/kw_max:.4f}")

# Steady-state peaks
ss_kw = kw_trace[Nt//2:]
ss_pkw = pkw_trace[Nt//2:]
kw_peaks = np.where((ss_kw[1:-1] > ss_kw[:-2]) & (ss_kw[1:-1] > ss_kw[2:]))[0]
pkw_peaks = np.where((ss_pkw[1:-1] > ss_pkw[:-2]) & (ss_pkw[1:-1] > ss_pkw[2:]))[0]
if len(kw_peaks) > 0 and len(pkw_peaks) > 0:
    print(f"\n  Steady-state k-wave  positive peaks (from Nt//2): {(kw_peaks[:5] + Nt//2).tolist()}")
    print(f"  Steady-state pykwav positive peaks (from Nt//2): {(pkw_peaks[:5] + Nt//2).tolist()}")
    T = np.diff(kw_peaks).mean() if len(kw_peaks) > 1 else 12.8
    print(f"  Period: {T:.2f} steps")
    if len(kw_peaks) > 0 and len(pkw_peaks) > 0:
        pk_kw = kw_peaks[0] + Nt//2
        pk_pkw = pkw_peaks[0] + Nt//2
        phase_steps = pk_pkw - pk_kw
        phase_deg = phase_steps / T * 360
        print(f"  Phase offset: {phase_steps} steps = {phase_deg:.1f} deg (pykwav - kwave)")
