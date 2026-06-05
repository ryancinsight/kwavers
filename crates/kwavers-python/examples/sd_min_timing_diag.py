#!/usr/bin/env python3
"""
Minimal timing diagnostic: runs just 20 time steps for a small grid.
Compares pressure at source point and one propagation point between
k-wave-python and pykwavers to isolate the 2-step phase advance.
"""
import sys, os
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()

import pykwavers as pkw
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.data import Vector
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

# Small grid for fast runs
N = 16
DX = 1e-3   # 1mm spacing
C0 = 1500.0
RHO0 = 1000.0
# dt from CFL=0.3
DT = 0.3 * DX / C0  # = 2e-7 s
NT = 20  # Just 20 steps to see timing

# Source at center
SX, SY, SZ = N//2, N//2, N//2  # [8,8,8] 0-indexed

print(f"Grid: {N}^3, dx={DX*1e3:.2f}mm, dt={DT:.3e}s, NT={NT}")
print(f"CFL = {C0*DT/DX:.3f}")
print(f"Source at [{SX},{SY},{SZ}] 0-indexed")

# Signal: impulse at step 1 (signal[0]=0, signal[1]=1 Pa, rest=0)
# This makes the source inject at step 1 only, giving a clean single-step injection.
signal_delta = np.zeros(NT)
signal_delta[1] = 1.0  # inject 1 Pa at step 1

# ─── k-wave-python ───────────────────────────────────────────────────────────
kgrid = kWaveGrid(Vector([N, N, N]), Vector([DX, DX, DX]))
kgrid.setTime(NT, DT)   # set exactly NT steps at DT
medium = kWaveMedium(sound_speed=C0)

src_mask = np.zeros((N, N, N), dtype=np.float64)
src_mask[SX, SY, SZ] = 1.0

# Sensor at source and at a propagation point 3 cells away
sensor_mask = np.zeros((N, N, N), dtype=np.int32)
sensor_mask[SX, SY, SZ] = 1        # source point
sensor_mask[SX-3, SY, SZ] = 1      # 3 cells toward -x

source_kw = kSource()
source_kw.p_mask = src_mask
source_kw.p = signal_delta.reshape(1, -1)   # (1, NT)

sensor_kw = kSensor(sensor_mask)

sim_opts = SimulationOptions(pml_size=2, data_cast="single", save_to_disk=True)
exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

print("\n[k-wave] Running...")
result_kw = kspaceFirstOrder3D(
    medium=medium, kgrid=kgrid,
    source=source_kw, sensor=sensor_kw,
    simulation_options=sim_opts,
    execution_options=exec_opts,
)

p_kw = np.asarray(result_kw["p"], dtype=np.float64)
print(f"  k-wave sensor_data shape: {p_kw.shape}")

# k-wave returns (n_sensors, NT) or (NT, n_sensors)
if p_kw.shape[0] == NT:
    p_kw = p_kw.T  # now (n_sensors, NT)

# Sensor ordering: find which row corresponds to which sensor
# In k-Wave Python, sensor data is ordered by find(sensor_mask) in Fortran order
# Fortran (column-major) find: z varies fastest, then y, then x
# So sensor at (SX-3,SY,SZ) and (SX,SY,SZ):
# Flatten index (Fortran order): idx = SX + N*(SY + N*SZ)
def fortran_flat(ix, iy, iz, n):
    return ix + n*(iy + n*iz)

f_src = fortran_flat(SX, SY, SZ, N)
f_prop = fortran_flat(SX-3, SY, SZ, N)
# All sensor flat indices in Fortran order, sorted
sensor_flat = sorted([f_src, f_prop])
row_src = sensor_flat.index(f_src)
row_prop = sensor_flat.index(f_prop)

p_src_kw = p_kw[row_src]
p_prop_kw = p_kw[row_prop]

print(f"  k-wave source point trace (step 0-19):")
print(f"    {np.round(p_src_kw, 6)}")
print(f"  k-wave prop point trace (step 0-19):")
print(f"    {np.round(p_prop_kw, 6)}")

first_nonzero_src_kw = np.where(np.abs(p_src_kw) > 1e-10)[0]
first_nonzero_prop_kw = np.where(np.abs(p_prop_kw) > 1e-10)[0]
print(f"  k-wave: source first nonzero at step {first_nonzero_src_kw[0] if len(first_nonzero_src_kw) else 'none'}")
print(f"  k-wave: prop first nonzero at step {first_nonzero_prop_kw[0] if len(first_nonzero_prop_kw) else 'none'}")

# ─── pykwavers ───────────────────────────────────────────────────────────────
bool_sensor = sensor_mask.astype(bool)

grid_pkw = pkw.Grid(N, N, N, DX, DX, DX)
medium_pkw = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
source_pkw = pkw.Source.from_mask(src_mask, signal_delta, 250e3, mode="additive")
sensor_pkw = pkw.Sensor.from_mask(bool_sensor)

sim_pkw = pkw.Simulation(grid_pkw, medium_pkw, source_pkw, sensor_pkw, solver=pkw.SolverType.PSTD)
sim_pkw.set_pml_size(2)
sim_pkw.set_pml_inside(True)

print("\n[pykwavers] Running...")
result_pkw = sim_pkw.run(time_steps=NT, dt=DT)

sd_pkw = np.asarray(result_pkw.sensor_data, dtype=np.float64)
print(f"  pykwavers sensor_data shape: {sd_pkw.shape}")

# pykwavers sensor ordering: Fortran order (same as k-Wave)
# Find row indices for our two sensor points
all_sensor_flat_pkw = np.where(bool_sensor.flatten(order='F'))[0]
row_src_pkw = np.where(all_sensor_flat_pkw == f_src)[0][0]
row_prop_pkw = np.where(all_sensor_flat_pkw == f_prop)[0][0]

p_src_pkw = sd_pkw[row_src_pkw]
p_prop_pkw = sd_pkw[row_prop_pkw]

print(f"  pykwavers source point trace (step 0-19):")
print(f"    {np.round(p_src_pkw, 6)}")
print(f"  pykwavers prop point trace (step 0-19):")
print(f"    {np.round(p_prop_pkw, 6)}")

first_nonzero_src_pkw = np.where(np.abs(p_src_pkw) > 1e-10)[0]
first_nonzero_prop_pkw = np.where(np.abs(p_prop_pkw) > 1e-10)[0]
print(f"  pykwavers: source first nonzero at step {first_nonzero_src_pkw[0] if len(first_nonzero_src_pkw) else 'none'}")
print(f"  pykwavers: prop first nonzero at step {first_nonzero_prop_pkw[0] if len(first_nonzero_prop_pkw) else 'none'}")

# ─── Comparison ──────────────────────────────────────────────────────────────
print("\n=== COMPARISON ===")
print("Step |  k-wave src  | pkwavers src |  k-wave prop | pkwavers prop")
print("-"*75)
for t in range(NT):
    print(f"  {t:2d} |  {p_src_kw[t]:+.6f} | {p_src_pkw[t]:+.6f} | {p_prop_kw[t]:+.6f} | {p_prop_pkw[t]:+.6f}")

# Find lag at source point
from scipy import signal as sp_sig
if len(first_nonzero_src_kw) and len(first_nonzero_src_pkw):
    lag_src = int(first_nonzero_src_pkw[0]) - int(first_nonzero_src_kw[0])
    print(f"\nSource point first-nonzero lag: pkwavers - kwave = {lag_src} steps")
if len(first_nonzero_prop_kw) and len(first_nonzero_prop_pkw):
    lag_prop = int(first_nonzero_prop_pkw[0]) - int(first_nonzero_prop_kw[0])
    print(f"Prop point first-nonzero lag: pkwavers - kwave = {lag_prop} steps")

print("\nDone.")
