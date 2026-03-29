#!/usr/bin/env python3
"""
Propagation step diagnostic.

Tests propagation of an injected pressure field (step 2 -> 3) to isolate
the root cause of the 1.6427x amplitude discrepancy.

Both simulators agree at step 2 (ratio=1.000) but differ at step 3
(kwavers / k-Wave = 1.6427 at source point).

This script:
1. Records full 3D field at steps 2 and 3 from both simulators
2. Manually replicates one free propagation step from p^2 using numpy
   with different operator conventions (kwavers vs k-Wave)
3. Identifies which numpy formulation matches each simulator
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
pml_size = 0
src_ix = N // 2  # 8
Nt = 3
signal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

kgrid_tmp = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid_tmp.makeTime(c0)
dt = float(kgrid_tmp.dt)
print(f"N={N}, dx={dx}, c0={c0}, dt={dt:.6e}")

# ===== k-Wave =====
print("\n--- k-Wave (pml_inside=False, pml_size=0) ---")
kgrid = kWaveGrid([N, N, N], [dx, dx, dx])
kgrid.setTime(Nt, dt)

src_mask = np.zeros((N, N, N))
src_mask[src_ix, src_ix, src_ix] = 1
kw_source = kSource()
kw_source.p_mask = src_mask
kw_source.p = signal.reshape(1, -1)
kw_source.p_mode = "additive"

sen_mask = np.ones((N, N, N), dtype=bool)
kw_sensor = kSensor(sen_mask.astype(float))
kw_sensor.record = ["p"]

kw_res = kspaceFirstOrder3D(
    medium=kWaveMedium(sound_speed=c0, density=rho0),
    kgrid=kgrid,
    source=kw_source,
    sensor=kw_sensor,
    simulation_options=SimulationOptions(
        pml_inside=False, pml_size=pml_size, data_cast="double", save_to_disk=True,
    ),
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False, delete_data=True, verbose_level=0),
)
kw_p_raw = np.array(kw_res["p"])
print(f"k-Wave shape: {kw_p_raw.shape}")
if kw_p_raw.shape[0] == Nt and kw_p_raw.shape[1] != Nt:
    kw_p_raw = kw_p_raw.T
elif kw_p_raw.shape[1] == Nt:
    pass

kw_step2 = kw_p_raw[:, 1].reshape(N, N, N)
kw_step3 = kw_p_raw[:, 2].reshape(N, N, N)

# ===== pykwavers =====
print("\n--- pykwavers (pml_size=0) ---")
kwa_grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
py_src_mask = np.zeros((N, N, N), dtype=np.float64)
py_src_mask[src_ix, src_ix, src_ix] = 1.0
kwa_source = kw.Source.from_mask(py_src_mask, signal.copy(), 1.0, mode="additive")
py_sen_mask = np.ones((N, N, N), dtype=bool)
kwa_sensor = kw.Sensor.from_mask(py_sen_mask)
kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source, kwa_sensor, solver=kw.SolverType.PSTD)
kwa_sim.set_pml_size(0)
kwa_sim.set_pml_inside(False)
kwa_res = kwa_sim.run(time_steps=Nt, dt=dt)
kwa_p_raw = np.array(kwa_res.sensor_data)
print(f"kwavers shape: {kwa_p_raw.shape}")

kwa_step2 = kwa_p_raw[:, 2].reshape(N, N, N)
kwa_step3 = kwa_p_raw[:, 3].reshape(N, N, N)

print(f"\n=== Simulator comparison ===")
print(f"Step 2: kw={kw_step2[src_ix,src_ix,src_ix]:.6e}, kwa={kwa_step2[src_ix,src_ix,src_ix]:.6e}, ratio={kwa_step2[src_ix,src_ix,src_ix]/kw_step2[src_ix,src_ix,src_ix]:.6f}")
print(f"Step 3: kw={kw_step3[src_ix,src_ix,src_ix]:.6e}, kwa={kwa_step3[src_ix,src_ix,src_ix]:.6e}, ratio={kwa_step3[src_ix,src_ix,src_ix]/kw_step3[src_ix,src_ix,src_ix]:.6f}")

# ===== Manual numpy propagation =====
print(f"\n===== Manual numpy propagation =====")

dk = 2.0 * np.pi / (N * dx)

def make_k1d_std(n):
    """Standard: [0,1,...,n/2,...,-(n/2-1),...,-1]*dk  (+Nyquist at idx n/2)"""
    k = np.zeros(n)
    for i in range(n):
        if i <= n//2:
            k[i] = i * dk
        else:
            k[i] = (i - n) * dk
    return k

def make_k1d_kwave(n):
    """k-Wave: ifftshift of [-n/2,...,n/2-1]*dk  (-Nyquist at idx n/2)"""
    return np.fft.ifftshift(np.arange(-n//2, n//2) * dk)

def unnorm_sinc(x):
    out = np.ones_like(x, dtype=float)
    m = np.abs(x) > 1e-14
    out[m] = np.sin(x[m]) / x[m]
    return out

def make_ops(k1d, zero_nyquist=False, n=N, dx_val=dx):
    """Build 3D kappa and 1D shift ops from 1D k vector."""
    kx3, ky3, kz3 = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k_mag = np.sqrt(kx3**2 + ky3**2 + kz3**2)
    kappa = unnorm_sinc(0.5 * c0 * dt * k_mag)

    # 1D shift ops
    def shift_1d(kv, sign):
        # sign=+1: pos (p->u), sign=-1: neg (u->rho)
        ops = 1j * kv * np.exp(sign * 1j * kv * dx_val * 0.5)
        if zero_nyquist and n % 2 == 0:
            ops[n//2] = 0.0
        return ops

    ddx_pos = shift_1d(k1d, +1)
    ddx_neg = shift_1d(k1d, -1)
    return kappa, ddx_pos, ddx_neg

def propagate_one_step(p2_field, kappa, ddx_pos, ddy_pos, ddz_pos,
                       ddx_neg, ddy_neg, ddz_neg):
    """
    One free PSTD step from (p^n, u^{n-1/2}=0).
    Returns p^{n+1}.
    """
    p_hat = np.fft.fftn(p2_field)

    # Velocity update: ux = -dt/rho0 * IFFT(ddx_pos[i] * kappa[i,j,k] * p_hat[i,j,k])
    # Vectorized: broadcast ddx_pos over j,k dimensions
    ddx_kap_p = ddx_pos[:, np.newaxis, np.newaxis] * kappa * p_hat  # (N,N,N)
    ddy_kap_p = ddy_pos[np.newaxis, :, np.newaxis] * kappa * p_hat
    ddz_kap_p = ddz_pos[np.newaxis, np.newaxis, :] * kappa * p_hat

    ux = np.real(np.fft.ifftn(ddx_kap_p)) * (-dt / rho0)
    uy = np.real(np.fft.ifftn(ddy_kap_p)) * (-dt / rho0)
    uz = np.real(np.fft.ifftn(ddz_kap_p)) * (-dt / rho0)

    # Density update from velocity divergence
    # dux/dx = IFFT(ddx_neg[i] * kappa[i,j,k] * FFT(ux)[i,j,k])
    ux_hat = np.fft.fftn(ux)
    uy_hat = np.fft.fftn(uy)
    uz_hat = np.fft.fftn(uz)

    duxdx = np.real(np.fft.ifftn(ddx_neg[:, np.newaxis, np.newaxis] * kappa * ux_hat))
    duydy = np.real(np.fft.ifftn(ddy_neg[np.newaxis, :, np.newaxis] * kappa * uy_hat))
    duzdz = np.real(np.fft.ifftn(ddz_neg[np.newaxis, np.newaxis, :] * kappa * uz_hat))

    # Starting density: rhox=rhoy=rhoz = p2/(3*c0^2)  (since p = c0^2 * 3 * rho_each)
    rho_each = p2_field / (3.0 * c0**2)

    rhox_new = rho_each - dt * rho0 * duxdx
    rhoy_new = rho_each - dt * rho0 * duydy
    rhoz_new = rho_each - dt * rho0 * duzdz

    return c0**2 * (rhox_new + rhoy_new + rhoz_new)

k1d_std   = make_k1d_std(N)
k1d_kwave = make_k1d_kwave(N)

kappa_std,   ddx_std_pos,   ddx_std_neg   = make_ops(k1d_std,   zero_nyquist=False)
kappa_std_z, ddx_std_z_pos, ddx_std_z_neg = make_ops(k1d_std,   zero_nyquist=True)
kappa_kw,    ddx_kw_pos,    ddx_kw_neg    = make_ops(k1d_kwave, zero_nyquist=False)
kappa_kw_z,  ddx_kw_z_pos,  ddx_kw_z_neg = make_ops(k1d_kwave, zero_nyquist=True)

print(f"\nddx_pos[8] (Nyquist bin):")
print(f"  std +Nyquist (no zero): {ddx_std_pos[8]:.6e}")
print(f"  std +Nyquist (zeroed):  {ddx_std_z_pos[8]:.6e}")
print(f"  kwave -Nyquist:         {ddx_kw_pos[8]:.6e}")

configs = [
    ("std+Nyquist, no zero",  kappa_std,   ddx_std_pos,   ddx_std_neg),
    ("std+Nyquist, zeroed",   kappa_std_z, ddx_std_z_pos, ddx_std_z_neg),
    ("kwave -Nyquist",        kappa_kw,    ddx_kw_pos,    ddx_kw_neg),
    ("kwave -Nyquist, zeroed",kappa_kw_z,  ddx_kw_z_pos,  ddx_kw_z_neg),
]

ref_kw  = kw_step3[src_ix, src_ix, src_ix]
ref_kwa = kwa_step3[src_ix, src_ix, src_ix]

print(f"\nStarting from k-Wave step2 field:")
print(f"{'Config':35s} {'src_val':>14} {'vs kwave':>10} {'vs kwavers':>12}")
for name, kap, dpos, dneg in configs:
    p3 = propagate_one_step(kw_step2, kap, dpos, dpos, dpos, dneg, dneg, dneg)
    v = p3[src_ix, src_ix, src_ix]
    print(f"{name:35s} {v:14.6e} {v/ref_kw:10.6f} {v/ref_kwa:12.6f}")

print(f"\nk-Wave step3 reference:  {ref_kw:.6e}")
print(f"kwavers step3 reference: {ref_kwa:.6e}")

print(f"\nStarting from kwavers step2 field:")
print(f"{'Config':35s} {'src_val':>14} {'vs kwave':>10} {'vs kwavers':>12}")
for name, kap, dpos, dneg in configs:
    p3 = propagate_one_step(kwa_step2, kap, dpos, dpos, dpos, dneg, dneg, dneg)
    v = p3[src_ix, src_ix, src_ix]
    print(f"{name:35s} {v:14.6e} {v/ref_kw:10.6f} {v/ref_kwa:12.6f}")

# Also test with NO kappa correction in propagation
print(f"\n--- No kappa correction in propagation ---")
kappa_ones = np.ones((N, N, N))
for name, _, dpos, dneg in [(c[0], None, c[2], c[3]) for c in configs[:2]]:
    p3 = propagate_one_step(kw_step2, kappa_ones, dpos, dpos, dpos, dneg, dneg, dneg)
    v = p3[src_ix, src_ix, src_ix]
    print(f"no-kappa {name:25s}: {v:.6e}  vs_kw={v/ref_kw:.6f}  vs_kwa={v/ref_kwa:.6f}")

# Check if kwavers matches a propagation with kappa=1 (no correction)
print(f"\n--- Summary ---")
print(f"k-Wave step3[src]: {ref_kw:.6e}")
print(f"kwavers step3[src]: {ref_kwa:.6e}")
print(f"Ratio kwavers/kwave: {ref_kwa/ref_kw:.6f}")
