"""
SD focussed detector sign diagnostic.
Runs pykwavers with a SINGLE sensor at the bowl apex [10,31,31]
to check whether the sign issue is in individual sensor points
or in the bowl summation.
"""
from __future__ import annotations
import numpy as np, sys, time
sys.path.insert(0, '.')
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()
import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_bowl

N = 64; DX = 100e-3 / N; C0 = 1500.0; RHO0 = 1000.0
SOURCE_FREQ = 0.25e6; PML_SIZE = 10

GRID_SIZE = Vector([N, N, N])
DX_VEC = Vector([DX, DX, DX])
SPHERE_OFFSET = 10; BOWL_RADIUS = N // 2; BOWL_DIAMETER = N // 2 + 1
BOWL_POS  = Vector([1 + SPHERE_OFFSET, N // 2, N // 2])  # 1-indexed
FOCUS_POS = Vector([N // 2, N // 2, N // 2])              # 1-indexed

kgrid  = kWaveGrid(GRID_SIZE, DX_VEC)
medium = kWaveMedium(sound_speed=C0)
kgrid.makeTime(medium.sound_speed)
dt = float(kgrid.dt); Nt = int(kgrid.Nt)
print(f"Grid: {N}^3, dx={DX*1e3:.4f}mm, dt={dt:.3e}s, Nt={Nt}")

# Build the filtered CW signal
raw_signal = np.sin(2 * np.pi * SOURCE_FREQ * kgrid.t_array)
filtered   = filter_time_series(kgrid, medium, raw_signal)
signal_1d  = np.asarray(filtered, dtype=np.float64).flatten()
print(f"Signal range: [{signal_1d.min():.4f}, {signal_1d.max():.4f}]")
print(f"Signal first non-zero indices: {np.where(np.abs(signal_1d) > 1e-6)[0][:5].tolist()}")

# Source at bowl focus (sphere center) [42,31,31] 0-indexed
SRC1_IX = int(SPHERE_OFFSET + BOWL_RADIUS)  # 42
SRC1_IY = N // 2 - 1                         # 31
SRC1_IZ = N // 2 - 1                         # 31
print(f"Source 0-idx: [{SRC1_IX}, {SRC1_IY}, {SRC1_IZ}]")

src_mask = np.zeros((N, N, N), dtype=np.float64)
src_mask[SRC1_IX, SRC1_IY, SRC1_IZ] = 1.0

grid   = pkw.Grid(N, N, N, DX, DX, DX)
med    = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
source = pkw.Source.from_mask(src_mask, signal_1d, SOURCE_FREQ, mode="additive")

# --- Test 1: Single sensor at bowl APEX [10,31,31] 0-indexed ---
apex_ix, apex_iy, apex_iz = SPHERE_OFFSET, N // 2 - 1, N // 2 - 1  # [10,31,31]
print(f"\nBowl apex 0-idx: [{apex_ix}, {apex_iy}, {apex_iz}]")
print(f"Distance source->apex = {abs(SRC1_IX - apex_ix)} dx = {abs(SRC1_IX - apex_ix)*DX*1e3:.1f}mm")
travel_steps = abs(SRC1_IX - apex_ix) * DX / C0 / dt
print(f"Expected travel time = {travel_steps:.1f} steps ({travel_steps*dt*1e6:.2f} µs)")

apex_mask = np.zeros((N, N, N), dtype=bool)
apex_mask[apex_ix, apex_iy, apex_iz] = True
sensor_apex = pkw.Sensor.from_mask(apex_mask)

sim1 = pkw.Simulation(grid, med, source, sensor_apex, solver=pkw.SolverType.PSTD)
sim1.set_pml_size(PML_SIZE)
sim1.set_pml_inside(True)

print("\n[Test 1] Running pykwavers with single sensor at bowl APEX...")
t0 = time.perf_counter()
r1 = sim1.run(time_steps=Nt, dt=dt)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
apex_trace = np.asarray(r1.sensor_data, dtype=np.float64).flatten()
arrival_idx = int(round(travel_steps))
print(f"  Apex trace at arrival [{arrival_idx-2}:{arrival_idx+6}]:")
print(f"    {apex_trace[max(0,arrival_idx-2):arrival_idx+6].round(6).tolist()}")
kw_max = np.abs(apex_trace).max()
first_nz = next((i for i in range(Nt) if abs(apex_trace[i]) > 0.01*kw_max), -1)
print(f"  First non-trivial sample: index={first_nz}, value={apex_trace[first_nz]:.6f}")
sign_apex = "POSITIVE (+)" if apex_trace[arrival_idx] >= 0 else "NEGATIVE (-)"
print(f"  Sign at arrival step {arrival_idx}: {sign_apex}")
print(f"  Peak: {kw_max:.6f} Pa at step {np.argmax(np.abs(apex_trace))}")
# Show steady-state phase
ss_trace = apex_trace[Nt//2:]
ss_peaks = np.where((ss_trace[1:-1] > ss_trace[:-2]) & (ss_trace[1:-1] > ss_trace[2:]))[0]
if len(ss_peaks) > 0:
    print(f"  Steady-state positive peaks at steps (from Nt//2): {(ss_peaks[:3] + Nt//2).tolist()}")

# --- Test 2: Full bowl sensor to compare with SD test ---
bowl_mask = make_bowl(GRID_SIZE, BOWL_POS, BOWL_RADIUS, BOWL_DIAMETER, FOCUS_POS)
bowl_mask_bool = np.asarray(bowl_mask, dtype=bool)
n_bowl = int(bowl_mask_bool.sum())
print(f"\n[Test 2] Bowl mask: {n_bowl} points")

sensor_bowl = pkw.Sensor.from_mask(bowl_mask_bool)
sim2 = pkw.Simulation(grid, med, source, sensor_bowl, solver=pkw.SolverType.PSTD)
sim2.set_pml_size(PML_SIZE)
sim2.set_pml_inside(True)

print("[Test 2] Running pykwavers with full BOWL sensor...")
t0 = time.perf_counter()
r2 = sim2.run(time_steps=Nt, dt=dt)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
sd2 = np.asarray(r2.sensor_data, dtype=np.float64)
print(f"  Sensor data shape: {sd2.shape}")

# Check a few individual sensor traces
for si in range(min(3, sd2.shape[0])):
    tr = sd2[si]
    print(f"  Sensor {si}: first_nz_idx={next((i for i in range(Nt) if abs(tr[i])>1e-6*np.abs(tr).max()), -1)}, "
          f"vals[{arrival_idx-1}:{arrival_idx+2}]={tr[arrival_idx-1:arrival_idx+2].round(6).tolist()}")

bowl_trace = np.sum(sd2, axis=0)
print(f"\n  BOWL SUM trace at [{arrival_idx-2}:{arrival_idx+6}]:")
print(f"    {bowl_trace[max(0,arrival_idx-2):arrival_idx+6].round(4).tolist()}")
print(f"  Bowl sum peak: {np.abs(bowl_trace).max():.4f} Pa")
first_nz_bowl = next((i for i in range(Nt) if abs(bowl_trace[i]) > 0.01*np.abs(bowl_trace).max()), -1)
print(f"  First non-trivial bowl sum: idx={first_nz_bowl}, val={bowl_trace[first_nz_bowl]:.4f}")
print(f"\n  Apex vs bowl sum at index {arrival_idx}: "
      f"apex={apex_trace[arrival_idx]:.6f}, bowl_sum={bowl_trace[arrival_idx]:.4f}")
