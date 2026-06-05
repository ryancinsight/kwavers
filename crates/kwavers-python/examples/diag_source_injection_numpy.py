#!/usr/bin/env python3
"""
Manual numpy implementation of PSTD source injection to diagnose differences.

Manually replicates:
  1. kwavers source injection (FFT → source_kappa → IFFT, standard FFT order k vectors)
  2. k-Wave source injection (as in kspaceFirstOrder3D.py: ifftshift(cos(...)) * fftn)

Compares with actual k-Wave C++ binary output to find root cause of leakage mismatch.
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
pml_size = 2
src_ix = N // 2  # 8

kgrid_tmp = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid_tmp.makeTime(c0)
dt = float(kgrid_tmp.dt)
print(f"N={N}, dx={dx}, c0={c0}, dt={dt:.6e}")

# Signal: step 2 uses signal=1.0
signal_val = 1.0
rho_scale = 2.0 * dt / (3 * c0 * dx)
print(f"rho_scale = {rho_scale:.6e}")

# ===== Build source_kappa two ways =====

# 1) kwavers style: cos(c0*dt*k_mag/2) with standard FFT order (positive Nyquist at index N//2)
k_idx = np.concatenate([np.arange(N//2 + 1), np.arange(-(N//2 - 1), 0)])  # [0,1,...,8,-7,...,-1]
k_1d_kwavers = k_idx * (2*np.pi / (N * dx))
kx, ky, kz = np.meshgrid(k_1d_kwavers, k_1d_kwavers, k_1d_kwavers, indexing='ij')
k_mag_kwavers = np.sqrt(kx**2 + ky**2 + kz**2)
source_kappa_kwavers = np.cos(0.5 * c0 * dt * k_mag_kwavers)

# 2) k-Wave style: ifftshift(cos(c0*k*dt/2)) where k is 3D magnitude in fftshift order
k_idx_fftshift = np.arange(-N//2, N//2)  # [-8,-7,...,0,...,7] for N=16
k_1d_kwave = k_idx_fftshift * (2*np.pi / (N * dx))
kx_f, ky_f, kz_f = np.meshgrid(k_1d_kwave, k_1d_kwave, k_1d_kwave, indexing='ij')
k_mag_kwave_fftshift = np.sqrt(kx_f**2 + ky_f**2 + kz_f**2)
source_kappa_kwave_fftshift = np.cos(0.5 * c0 * dt * k_mag_kwave_fftshift)
source_kappa_kwave = np.fft.ifftshift(source_kappa_kwave_fftshift)

print(f"\nSource kappa comparison:")
print(f"kwavers source_kappa at [0,0,0]: {source_kappa_kwavers[0,0,0]:.6f} (expect 1.0)")
print(f"k-Wave source_kappa at [0,0,0]: {source_kappa_kwave[0,0,0]:.6f} (expect 1.0)")
print(f"kwavers source_kappa at [{N//2},{N//2},{N//2}]: {source_kappa_kwavers[N//2,N//2,N//2]:.6f}")
print(f"k-Wave source_kappa at [{N//2},{N//2},{N//2}]: {source_kappa_kwave[N//2,N//2,N//2]:.6f}")
print(f"Max diff between kwavers and k-Wave source_kappa: {np.abs(source_kappa_kwavers - source_kappa_kwave).max():.6e}")
print(f"Are they identical? {np.allclose(source_kappa_kwavers, source_kappa_kwave)}")

# Show the 1D source_kappa along k_x axis for both
print(f"\n1D source_kappa along kx axis at ky=kz=0:")
print(f"{'idx':>4} {'kx_kwavers':>12} {'kx_kwave':>12} {'sk_kwavers':>12} {'sk_kwave':>12}")
for i in range(N):
    print(f"{i:4d} {k_1d_kwavers[i]:12.2f} {k_1d_kwave[i if i < N//2 else i] + (2*np.pi/(N*dx)) * (N//2 if i >= N//2 else 0):12.2f} "
          f"{source_kappa_kwavers[i,0,0]:12.6f} {source_kappa_kwave[i,0,0]:12.6f}")

# ===== Manual source injection =====

delta = np.zeros((N, N, N))
delta[src_ix, src_ix, src_ix] = signal_val * rho_scale

# kwavers method
dpx_fft = np.fft.fftn(delta)
dpx_filtered_kwavers = np.real(np.fft.ifftn(source_kappa_kwavers * dpx_fft))

# k-Wave method
dpx_filtered_kwave = np.real(np.fft.ifftn(source_kappa_kwave * dpx_fft))

# pressure = c0^2 * 3 * filtered_density (since added to all 3 components with density_scale=1)
p_kwavers_manual = c0**2 * 3 * dpx_filtered_kwavers
p_kwave_manual = c0**2 * 3 * dpx_filtered_kwave

print(f"\n===== Manual source injection results =====")
print(f"Source point:")
print(f"  kwavers manual: {p_kwavers_manual[src_ix,src_ix,src_ix]:.6e} Pa")
print(f"  k-Wave manual:  {p_kwave_manual[src_ix,src_ix,src_ix]:.6e} Pa")

# Analytical
mean_sk = source_kappa_kwave.mean()
p_expected = (2*c0*dt/dx) * signal_val * mean_sk
print(f"  Analytical: {p_expected:.6e} Pa (mean_sk={mean_sk:.4f})")

print(f"\n1D pressure along x-axis (y={src_ix}, z={src_ix}):")
print(f"{'x':>4} {'kwavers_manual':>16} {'kwave_manual':>14} {'ratio':>8}")
for x in range(N):
    v_kwa = p_kwavers_manual[x, src_ix, src_ix]
    v_kw = p_kwave_manual[x, src_ix, src_ix]
    ratio = v_kwa/v_kw if abs(v_kw) > 1e-30 else float('nan')
    print(f"{x:4d} {v_kwa:16.6e} {v_kw:14.6e} {ratio:8.3f}")

# ===== Get actual k-Wave binary result =====
print(f"\n===== Actual k-Wave binary (single step) =====")
kgrid = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid.setTime(2, dt)

src_mask = np.zeros((N, N, N))
src_mask[src_ix, src_ix, src_ix] = 1
kw_source = kSource()
kw_source.p_mask = src_mask
kw_source.p = np.array([[0.0, signal_val]])
kw_source.p_mode = "additive"

sen_mask_line = np.zeros((N, N, N), dtype=float)
for x in range(N):
    sen_mask_line[x, src_ix, src_ix] = 1
kw_sensor = kSensor(sen_mask_line)
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
kw_p_raw = np.array(kw_res["p"])
if kw_p_raw.shape[1] == 2:
    pass  # already (n_sensors, Nt)
else:
    kw_p_raw = kw_p_raw.T
kw_line = kw_p_raw[:, 1]  # step 2 (signal[1]=1.0)

print(f"\n{'x':>4} {'kwavers_manual':>16} {'kwave_manual':>14} {'kwave_binary':>14} {'ratio_kwa':>10} {'ratio_kw':>10}")
for x in range(N):
    v_kwa = p_kwavers_manual[x, src_ix, src_ix]
    v_kw = p_kwave_manual[x, src_ix, src_ix]
    v_bin = kw_line[x]
    r_kwa = v_kwa/v_bin if abs(v_bin) > 1e-30 else float('nan')
    r_kw = v_kw/v_bin if abs(v_bin) > 1e-30 else float('nan')
    print(f"{x:4d} {v_kwa:16.6e} {v_kw:14.6e} {v_bin:14.6e} {r_kwa:10.3f} {r_kw:10.3f}")
