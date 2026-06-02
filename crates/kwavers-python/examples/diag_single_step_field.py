#!/usr/bin/env python3
"""
Single-step full-field diagnostic.

Runs both simulators for exactly 1 step with a non-zero source, then
compares the FULL 3D pressure field (not just sensor values) to isolate
source injection differences.

Uses a small 16x16x16 grid for fast execution and easy analysis.
"""
import numpy as np
import sys

try:
    import pykwavers as kw
except ImportError:
    print("ERROR: pykwavers not found"); sys.exit(1)

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
except ImportError:
    print("ERROR: k-wave-python not found"); sys.exit(1)

N = 16
dx = 1e-3
c0 = 1500.0
rho0 = 1000.0
pml_size = 2  # small PML

kgrid_tmp = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid_tmp.makeTime(c0)
dt = float(kgrid_tmp.dt)
print(f"N={N}, dx={dx}, c0={c0}, dt={dt:.4e}")

# Use a simple signal: just 2 time steps [0.0, 1.0] so step 1 has no source
# and step 2 has signal=1.0
Nt = 2
signal = np.array([0.0, 1.0], dtype=np.float64)
print(f"Signal: {signal}")

src_ix = N // 2  # 8

# ===== k-Wave =====
print("\n--- k-Wave ---")
kgrid = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid.setTime(Nt, dt)

src_mask = np.zeros((N, N, N))
src_mask[src_ix, src_ix, src_ix] = 1
kw_source = kSource()
kw_source.p_mask = src_mask
kw_source.p = signal.reshape(1, -1)
kw_source.p_mode = "additive"

# Record ALL points as sensors
sen_mask = np.ones((N, N, N), dtype=bool)
kw_sensor = kSensor(sen_mask.astype(float))
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
# k-Wave returns (Nt, n_sensors) where n_sensors = N^3
kw_p_raw = np.array(kw_res["p"])  # shape (Nt, N^3) or (N^3, Nt)
print(f"k-Wave raw shape: {kw_p_raw.shape}")
# k-Wave returns (Nt, n_sensors) — always transpose to (n_sensors, Nt)
# when Nt < n_sensors, shape[0] = Nt < shape[1] = n_sensors
if kw_p_raw.shape[1] == Nt:
    pass  # already (n_sensors, Nt)
else:
    kw_p_raw = kw_p_raw.T  # (Nt, n_sensors) → (n_sensors, Nt)
print(f"k-Wave shape after transpose: {kw_p_raw.shape}")

# Step 2 field (index 1, 0-based): after step 2 which uses signal[1]=1.0
kw_field_step2 = kw_p_raw[:, 1].reshape(N, N, N)

# ===== pykwavers =====
print("\n--- pykwavers ---")
kwa_grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

py_src_mask = np.zeros((N, N, N), dtype=np.float64)
py_src_mask[src_ix, src_ix, src_ix] = 1.0
kwa_source = kw.Source.from_mask(py_src_mask, signal.copy(), 1.0, mode="additive")

py_sen_mask = np.ones((N, N, N), dtype=bool)
kwa_sensor = kw.Sensor.from_mask(py_sen_mask)

kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source, kwa_sensor, solver=kw.SolverType.PSTD)
kwa_sim.set_pml_size(pml_size)
kwa_sim.set_pml_inside(True)
kwa_res = kwa_sim.run(time_steps=Nt, dt=dt)
kwa_p_raw = np.array(kwa_res.sensor_data)  # shape (n_sensors, Nt+1) with t=0 initial
print(f"kwavers raw shape: {kwa_p_raw.shape}")

# kwavers: t=0 (zeros), t=1 (after step 1), t=2 (after step 2)
kwa_field_step2 = kwa_p_raw[:, 2].reshape(N, N, N)

# ===== Compare =====
print("\n===== COMPARISON: Source injection field at step 2 =====")
src = (src_ix, src_ix, src_ix)

kw_val_src = kw_field_step2[src]
kwa_val_src = kwa_field_step2[src]
print(f"Source point: k-Wave={kw_val_src:.6e}, kwavers={kwa_val_src:.6e}, ratio={kwa_val_src/kw_val_src:.4f}")

# Compute analytical expected value
# At step 2, signal=1.0. Injection = FFT→source_kappa→IFFT
# At source point: p_expected = (2*c0*dt/dx) * signal * mean(source_kappa)
# For 16³ grid, source_kappa = cos(c_ref * dt * k_mag / 2)
dk = 2*np.pi / (N * dx)
k_vals = np.concatenate([np.arange(N//2+1), np.arange(-(N//2-1), 0)]) * dk
kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
source_kappa_arr = np.cos(0.5 * c0 * dt * k_mag)
mean_sk = source_kappa_arr.mean()
p_expected = (2*c0*dt/dx) * 1.0 * mean_sk
print(f"Analytical: mean(source_kappa)={mean_sk:.4f}, expected_p_src={p_expected:.6e}")

# Show off-source leakage pattern along x-axis (y=src, z=src)
print(f"\n--- Pressure along x-axis at step 2 (y={src_ix}, z={src_ix}) ---")
print(f"{'x':>4} {'k-Wave':>12} {'kwavers':>12} {'ratio':>8}")
for x in range(N):
    kw_v = kw_field_step2[x, src_ix, src_ix]
    kwa_v = kwa_field_step2[x, src_ix, src_ix]
    ratio = kwa_v/kw_v if abs(kw_v) > 1e-30 else float('nan')
    print(f"{x:4d} {kw_v:12.4e} {kwa_v:12.4e} {ratio:8.3f}")

# Statistics
print(f"\n--- Field Statistics ---")
print(f"k-Wave: max={np.abs(kw_field_step2).max():.4e}, mean={np.abs(kw_field_step2).mean():.4e}")
print(f"kwavers: max={np.abs(kwa_field_step2).max():.4e}, mean={np.abs(kwa_field_step2).mean():.4e}")
print(f"Max abs diff: {np.abs(kw_field_step2 - kwa_field_step2).max():.4e}")
print(f"RMS diff: {np.sqrt(np.mean((kw_field_step2 - kwa_field_step2)**2)):.4e}")

# Also look at field after step 1 (should be all zeros since signal[0]=0)
kw_field_step1 = kw_p_raw[:, 0].reshape(N, N, N)
kwa_field_step1 = kwa_p_raw[:, 1].reshape(N, N, N)
print(f"\n--- Step 1 (signal[0]=0) ---")
print(f"k-Wave max: {np.abs(kw_field_step1).max():.4e}")
print(f"kwavers max: {np.abs(kwa_field_step1).max():.4e}")
