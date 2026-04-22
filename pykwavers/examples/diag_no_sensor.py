#!/usr/bin/env python3
"""
Minimal diagnostic: run pykwavers with single-point sensor to check propagation ratio.
Target ratio: p[step3] / p[step2] ~ 0.211 (from Rust unit test reference)
"""
import sys, os
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_parity_utils import bootstrap_example_paths
bootstrap_example_paths()

import pykwavers as pkw

N = 16
DX = 1e-3
C0 = 1500.0
RHO0 = 1000.0
DT = 0.3 * DX / C0  # ~2e-7 s
NT = 4  # exactly 4 steps to match Rust unit test

SX, SY, SZ = N//2, N//2, N//2

print(f"Grid: {N}^3, dx={DX*1e3:.2f}mm, dt={DT:.3e}s, NT={NT}")
print(f"Source at [{SX},{SY},{SZ}] 0-indexed")

signal = np.zeros(NT)
signal[1] = 1.0  # inject 1 Pa at step 1

src_mask = np.zeros((N, N, N), dtype=np.float64)
src_mask[SX, SY, SZ] = 1.0

grid   = pkw.Grid(N, N, N, DX, DX, DX)
medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
source = pkw.Source.from_mask(src_mask, signal, 250e3, mode="additive")

# ─── Test A: sensor at source point only ─────────────────────────────────────
print("\n[Test A] Sensor at [8,8,8] only, CPML=2")
smask_a = np.zeros((N, N, N), dtype=bool)
smask_a[SX, SY, SZ] = True
sensor_a = pkw.Sensor.from_mask(smask_a)

sim_a = pkw.Simulation(grid, medium, source, sensor_a, solver=pkw.SolverType.PSTD)
sim_a.set_pml_size(2)
result_a = sim_a.run(time_steps=NT, dt=DT)

sd_a = np.asarray(result_a.sensor_data, dtype=np.float64)
print(f"  sensor_data shape: {sd_a.shape}")
if sd_a.shape[0] > 0 and sd_a.shape[1] >= 4:
    p = sd_a[0]
    print(f"  p[8,8,8] trace: {np.round(p, 6)}")
    print(f"  step1={p[0]:.6e}  step2={p[1]:.6e}  step3={p[2]:.6e}  step4={p[3]:.6e}")
    if p[1] > 0:
        ratio = p[2] / p[1]
        print(f"  ratio step3/step2 = {ratio:.4f}  (target: ~0.211)")
        if abs(ratio - 0.211) < 0.01:
            print("  [PASS]")
        else:
            print(f"  [FAIL] expected ~0.211, got {ratio:.4f}")

# ─── Test B: TWO sensors (source + propagation) ───────────────────────────────
print("\n[Test B] Two sensors: [8,8,8] and [5,8,8], CPML=2")
smask_b = np.zeros((N, N, N), dtype=bool)
smask_b[SX, SY, SZ] = True
smask_b[SX-3, SY, SZ] = True
sensor_b = pkw.Sensor.from_mask(smask_b)

sim_b = pkw.Simulation(grid, medium, source, sensor_b, solver=pkw.SolverType.PSTD)
sim_b.set_pml_size(2)
result_b = sim_b.run(time_steps=NT, dt=DT)

sd_b = np.asarray(result_b.sensor_data, dtype=np.float64)
print(f"  sensor_data shape: {sd_b.shape}")

# Find source row using Fortran flat ordering
f_src  = SX   + N*(SY + N*SZ)
f_prop = (SX-3) + N*(SY + N*SZ)
all_flat = np.where(smask_b.flatten(order='F'))[0]
print(f"  Fortran-flat indices: f_src={f_src}, f_prop={f_prop}, sorted={sorted([f_src, f_prop])}")
row_src  = list(sorted([f_src, f_prop])).index(f_src)
row_prop = list(sorted([f_src, f_prop])).index(f_prop)
print(f"  row_src={row_src}, row_prop={row_prop}")

if sd_b.shape[0] > row_src and sd_b.shape[1] >= 4:
    p_src = sd_b[row_src]
    print(f"  p[8,8,8] trace: {np.round(p_src, 6)}")
    if p_src[1] > 0:
        ratio = p_src[2] / p_src[1]
        print(f"  ratio step3/step2 = {ratio:.4f}  (target: ~0.211)")

# ─── Test C: sensor far from source, CPML=2 ──────────────────────────────────
print("\n[Test C] Sensor at off-center [2,8,8] only, CPML=2")
smask_c = np.zeros((N, N, N), dtype=bool)
smask_c[2, SY, SZ] = True
sensor_c = pkw.Sensor.from_mask(smask_c)

sim_c = pkw.Simulation(grid, medium, source, sensor_c, solver=pkw.SolverType.PSTD)
sim_c.set_pml_size(2)
result_c = sim_c.run(time_steps=NT, dt=DT)
sd_c = np.asarray(result_c.sensor_data, dtype=np.float64)
print(f"  sensor_data shape: {sd_c.shape}")

# Regardless of sensor placement, test with direct field access via separate run
# We do this by also running Test A simultaneously to get p[8,8,8]
# (sensor reads the same underlying field regardless of which points are in sensor)

print("\nDone.")
