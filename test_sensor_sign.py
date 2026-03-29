#!/usr/bin/env python3
"""
Quick diagnostic: test whether sensor data from pykwavers has correct sign at off-source positions.
Directly inspects pressure field values by bypassing sensor and comparing.
"""
import numpy as np
import sys
sys.path.insert(0, 'pykwavers/python')
import pykwavers as kw

N = 16
dx = 1e-3
c0 = 1500.0
rho0 = 1000.0
src_ix = N // 2  # 8

# Compute dt from CFL condition
dt = 0.3 * dx / c0
print(f"N={N}, dx={dx}, c0={c0}, dt={dt:.6e}")
print(f"Source at [{src_ix},{src_ix},{src_ix}]")

Nt = 2
signal = np.array([0.0, 1.0], dtype=np.float64)

kwa_grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

# Full-grid sensor mask
py_sen_mask = np.ones((N, N, N), dtype=bool)
kwa_sensor = kw.Sensor.from_mask(py_sen_mask)

py_src_mask = np.zeros((N, N, N), dtype=np.float64)
py_src_mask[src_ix, src_ix, src_ix] = 1.0
kwa_source = kw.Source.from_mask(py_src_mask, signal.copy(), 1.0, mode="additive")

kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source, kwa_sensor, solver=kw.SolverType.PSTD)
kwa_sim.set_pml_size(0)  # No PML for clean test
kwa_sim.set_pml_inside(True)
kwa_res = kwa_sim.run(time_steps=Nt, dt=dt)
kwa_p_raw = np.array(kwa_res.sensor_data)  # (n_sensors, Nt+1)
print(f"kwavers raw shape: {kwa_p_raw.shape}")

# kwavers: t=0 (zeros), t=1 (after step 1), t=2 (after step 2)
kwa_field_c_order = kwa_p_raw[:, 2].reshape(N, N, N)  # C-order reshape

# The sensor is in Fortran order (x fastest), so C-order reshape gives:
# kwa_field_c_order[k, j, i] = p_actual[i, j, k]
# i.e., kwa_field_c_order[0, 8, 8] = p_actual[8, 8, 0] (z=0 position)
# and   kwa_field_c_order[8, 8, 0] = p_actual[0, 8, 8] (x=0 position)

print(f"\n=== C-order reshape (Fortran->C transposition) ===")
print(f"kwa_field_c_order[{src_ix},{src_ix},{src_ix}] = {kwa_field_c_order[src_ix,src_ix,src_ix]:.6e} (source position)")
print(f"kwa_field_c_order[0,{src_ix},{src_ix}] = {kwa_field_c_order[0,src_ix,src_ix]:.6e} -> actual p[{src_ix},{src_ix},0] (z-axis)")
print(f"kwa_field_c_order[{src_ix},{src_ix},0] = {kwa_field_c_order[src_ix,src_ix,0]:.6e} -> actual p[0,{src_ix},{src_ix}] (x-axis)")

# Correct interpretation: Fortran-order → use Fortran reshape
kwa_field_fortran = kwa_p_raw[:, 2].reshape(N, N, N, order='F')
# In Fortran-order reshape: kwa_field_fortran[i, j, k] = p_actual[i, j, k]
print(f"\n=== Fortran-order reshape (correct for Fortran sensor data) ===")
print(f"kwa_field_fortran[{src_ix},{src_ix},{src_ix}] = {kwa_field_fortran[src_ix,src_ix,src_ix]:.6e} (source position)")
print(f"kwa_field_fortran[0,{src_ix},{src_ix}] = {kwa_field_fortran[0,src_ix,src_ix]:.6e} -> actual p[0,{src_ix},{src_ix}] (x-axis, should be negative)")
print(f"kwa_field_fortran[{src_ix},{src_ix},0] = {kwa_field_fortran[src_ix,src_ix,0]:.6e} -> actual p[{src_ix},{src_ix},0] (z-axis, should be negative)")

# Analytical formula: at source point, p ≈ c0^2 * 3 * rho_scale * mean(source_kappa)
rho_scale = 2.0 * dt / (3 * c0 * dx)
dk = 2*np.pi / (N * dx)
k_idx = np.concatenate([np.arange(N//2 + 1), np.arange(-(N//2 - 1), 0)])
k_1d = k_idx * dk
kx, ky, kz = np.meshgrid(k_1d, k_1d, k_1d, indexing='ij')
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
source_kappa = np.cos(0.5 * c0 * dt * k_mag)
mean_sk = source_kappa.mean()
p_expected = c0**2 * 3 * rho_scale * mean_sk
print(f"\n=== Analytical reference ===")
print(f"rho_scale = {rho_scale:.6e}")
print(f"mean(source_kappa) = {mean_sk:.6f}")
print(f"Expected p at source = {p_expected:.6e} Pa")
print(f"\nExpected sign at [0,8,8] (x-axis): NEGATIVE (k-Wave reference: -4.89e-4 Pa)")

# Check sign at source
p_src = kwa_field_fortran[src_ix, src_ix, src_ix]
p_off_x = kwa_field_fortran[0, src_ix, src_ix]

print(f"\n=== SIGN CHECK (Fortran reshape) ===")
print(f"p[src] = {p_src:.6e}: {'OK (positive)' if p_src > 0 else 'WRONG (should be positive)'}")
print(f"p[0,8,8] = {p_off_x:.6e}: {'OK (negative)' if p_off_x < 0 else 'WRONG (should be negative)'}")
