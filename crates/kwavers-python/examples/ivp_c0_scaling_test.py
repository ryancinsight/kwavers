#!/usr/bin/env python3
"""
ivp_c0_scaling_test.py
======================
Test how c_eff_apparent scales with c0, DX, and NX.
If c_eff/c0 is constant: the wave speed multiplier is intrinsic.
If c_eff changes differently: we can triangulate the actual bug.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
_pyp  = _root / "crates" / "kwavers-python" / "python"
if str(_pyp) not in sys.path:
    sys.path.insert(0, str(_pyp))

import pykwavers as pkw

def sinc_u(x):
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x)/x)

def run_one_step(NX, DX, C0, RHO0, DT, sigma=2.5):
    """Run pykwavers 1 step, return 1D FFT ratio at k1."""
    NY = NZ = NX
    cx, cy, cz = NX//2, NY//2, NZ//2
    x_arr = np.arange(NX)
    p0 = np.exp(-((x_arr - cx)**2)/(2*sigma**2))[:, None, None] * np.ones((NX, NY, NZ))

    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_initial_pressure(p0.copy())
    sensor_mask = np.ones((NX, NY, NZ), dtype=bool)
    sensor = pkw.Sensor.from_mask(sensor_mask)
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(0)
    result = sim.run(time_steps=1, dt=DT)
    data = np.asarray(result.sensor_data, dtype=np.float64)
    p1 = data[:, 0].reshape(NX, NY, NZ, order='F')

    p0_x = p0[:, cy, cz]
    p1_x = p1[:, cy, cz]
    P0 = np.fft.fft(p0_x)
    P1 = np.fft.fft(p1_x)
    ratio_k1 = abs(P1[1]) / (abs(P0[1]) + 1e-30)

    dk = 2*np.pi/(NX*DX)
    k1 = dk  # = 2pi/L
    # naive c_eff from (1-ratio)/dt²/k1²
    c_eff_naive = np.sqrt((1-ratio_k1)/(DT**2 * k1**2)) if (1-ratio_k1)>0 else float('nan')
    # corrected: with source_kappa, (1-ratio) ≈ (9/8)*c0²*dt²*k1²  → c_eff_corr = c0*sqrt(8/9)*(1-ratio)^0.5/(dt*k1)
    c_eff_corrected = np.sqrt((1-ratio_k1)/(DT**2 * k1**2) * 8/9)

    return ratio_k1, c_eff_naive, c_eff_corrected, k1

DT = 2e-9
RHO0 = 1000.0
NX_base = 16
DX_base = 1e-3/64  # 15.625 um

print("=== C0 scaling test (NX=16, DX=15.625 um, DT=2ns) ===")
print(f"{'C0 (m/s)':>12}  {'ratio_k1':>10}  {'c_eff (m/s)':>12}  {'c_eff/c0':>10}  {'c_eff_corr':>12}  {'corr/c0':>10}")
for C0 in [750, 1000, 1200, 1500, 2000, 3000]:
    ratio_k1, c_eff, c_corr, k1 = run_one_step(NX_base, DX_base, C0, RHO0, DT)
    print(f"{C0:12.0f}  {ratio_k1:10.6f}  {c_eff:12.2f}  {c_eff/C0:10.4f}  {c_corr:12.2f}  {c_corr/C0:10.4f}")

print()
print("=== DX scaling test (NX=16, C0=1500, DT=2ns) ===")
print(f"{'DX (um)':>10}  {'k1 (rad/m)':>12}  {'ratio_k1':>10}  {'c_eff (m/s)':>12}  {'c_eff/c0':>10}")
C0 = 1500.0
for DX_factor in [0.5, 1.0, 2.0, 4.0]:
    DX = DX_base * DX_factor
    ratio_k1, c_eff, c_corr, k1 = run_one_step(NX_base, DX, C0, RHO0, DT)
    print(f"{DX*1e6:10.3f}  {k1:12.2f}  {ratio_k1:10.6f}  {c_eff:12.2f}  {c_eff/C0:10.4f}")

print()
print("=== NX scaling test (DX=15.625 um, C0=1500, DT=2ns) ===")
print(f"{'NX':>6}  {'k1 (rad/m)':>12}  {'ratio_k1':>10}  {'c_eff (m/s)':>12}  {'c_eff/c0':>10}")
DX = DX_base
C0 = 1500.0
for NX in [8, 16, 32, 64]:
    ratio_k1, c_eff, c_corr, k1 = run_one_step(NX, DX, C0, RHO0, DT)
    print(f"{NX:6d}  {k1:12.2f}  {ratio_k1:10.6f}  {c_eff:12.2f}  {c_eff/C0:10.4f}")

print()
print("=== DT scaling test (NX=16, DX=15.625 um, C0=1500) ===")
print(f"{'DT (ns)':>10}  {'theta=c0*dt*k1/2':>18}  {'ratio_k1':>10}  {'c_eff/c0':>10}  {'pred_ratio':>12}  {'obs/pred':>10}")
NX = 16; DX = DX_base; C0 = 1500.0
dk = 2*np.pi/(NX*DX)
k1 = dk
for DT_ns in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
    DT2 = DT_ns*1e-9
    theta = C0*DT2*k1/2
    pred_ratio = np.cos(theta)*(1-4*np.sin(theta)**2)
    ratio_k1, c_eff, c_corr, _ = run_one_step(NX, DX, C0, RHO0, DT2)
    print(f"{DT_ns:10.1f}  {theta:18.6f}  {ratio_k1:10.6f}  {c_eff/C0:10.4f}  {pred_ratio:12.6f}  {ratio_k1/pred_ratio:10.6f}")

print()
print("=== Pure Python c_eff check ===")
DT = 2e-9; C0 = 1500.0; NX = 16; DX = DX_base
dk = 2*np.pi/(NX*DX); k1 = dk
kx_1d = np.fft.fftfreq(NX, d=DX)*2*np.pi; ky_1d = np.fft.fftfreq(NX, d=DX)*2*np.pi; kz_1d = np.fft.fftfreq(NX, d=DX)*2*np.pi
KX, KY, KZ = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
KMAG = np.sqrt(KX**2+KY**2+KZ**2)
kappa = sinc_u(C0*DT*KMAG/2.0)
DDX_P = (1j*kx_1d*np.exp(+1j*kx_1d*DX/2))[:,None,None]
DDX_N = (1j*kx_1d*np.exp(-1j*kx_1d*DX/2))[:,None,None]
DDY_P = (1j*ky_1d*np.exp(+1j*ky_1d*DX/2))[None,:,None]
DDY_N = (1j*ky_1d*np.exp(-1j*ky_1d*DX/2))[None,:,None]
DDZ_P = (1j*kz_1d*np.exp(+1j*kz_1d*DX/2))[None,None,:]
DDZ_N = (1j*kz_1d*np.exp(-1j*kz_1d*DX/2))[None,None,:]
cx=cy=cz=NX//2; sigma=2.5
x_arr = np.arange(NX)
p0 = np.exp(-((x_arr-cx)**2)/(2*sigma**2))[:,None,None]*np.ones((NX,NX,NX))
ux=np.zeros((NX,NX,NX)); uy=np.zeros((NX,NX,NX)); uz=np.zeros((NX,NX,NX))
rhox=p0/(3*C0**2); rhoy=p0/(3*C0**2); rhoz=p0/(3*C0**2)
pk=np.fft.fftn(p0)
ux -= DT/RHO0*np.real(np.fft.ifftn(kappa*DDX_P*pk))
uy -= DT/RHO0*np.real(np.fft.ifftn(kappa*DDY_P*pk))
uz -= DT/RHO0*np.real(np.fft.ifftn(kappa*DDZ_P*pk))
rhox -= DT*RHO0*np.real(np.fft.ifftn(kappa*DDX_N*np.fft.fftn(ux)))
rhoy -= DT*RHO0*np.real(np.fft.ifftn(kappa*DDY_N*np.fft.fftn(uy)))
rhoz -= DT*RHO0*np.real(np.fft.ifftn(kappa*DDZ_N*np.fft.fftn(uz)))
p1_py = C0**2*(rhox+rhoy+rhoz)
P0_x = np.fft.fft(p0[:,cy,cz])
P1_x = np.fft.fft(p1_py[:,cy,cz])
ratio_py = abs(P1_x[1])/(abs(P0_x[1])+1e-30)
c_eff_py = np.sqrt((1-ratio_py)/(DT**2*k1**2))
print(f"Pure Python: ratio_k1={ratio_py:.6f}  c_eff={c_eff_py:.2f} m/s  c_eff/c0={c_eff_py/C0:.4f}")
theta = C0*DT*k1/2
pred = 1-4*np.sin(theta)**2
print(f"  Predicted ratio (no sk): {pred:.6f}  diff={ratio_py-pred:.2e}")
