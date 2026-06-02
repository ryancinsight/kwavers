#!/usr/bin/env python3
"""Diagnostic: reference 1D elastic FD simulation to compare with Rust output."""
import numpy as np

NX=40; DX=0.5e-3; CP=3000.0; CS=1500.0; RHO=1900.0
SIGMA_CELLS=8; CFL=0.3; DT=5e-8; NT=110; PML=8
SRC_X=14; SX_NEAR=24; SX_FAR=29

mu = RHO * CS**2
lam = RHO*(CP**2 - 2*CS**2)

t = np.arange(NT)*DT; t0=5*SIGMA_CELLS*DT; sigma_s=SIGMA_CELLS*DT
tau=(t-t0)/sigma_s; signal=(1-tau**2)*np.exp(-0.5*tau**2)

ux = np.zeros(NX); vx = np.zeros(NX)

# PML: sigma_max via Collino & Tsogka 2001 optimized formula
sigma_max = -np.log(1e-4)*CP / (2*PML*DX)
sigma_pml = np.zeros(NX)
for i in range(PML):
    s = sigma_max * ((PML-i-1)/(PML-1))**2
    sigma_pml[i] = s
    sigma_pml[NX-1-i] = s

def fd4_x(f):
    d = np.zeros(NX)
    for i in range(2, NX-2):
        d[i] = (-f[i+2]+8*f[i+1]-8*f[i-1]+f[i-2])/(12*DX)
    d[0] = (f[1]-f[0])/DX
    d[1] = (f[2]-f[0])/(2*DX)
    d[NX-2] = (f[NX-1]-f[NX-3])/(2*DX)
    d[NX-1] = (f[NX-1]-f[NX-2])/DX
    return d

near_sig=np.zeros(NT); far_sig=np.zeros(NT)

for step in range(NT):
    vx[SRC_X] += signal[step]

    stress_xx = (lam+2*mu)*fd4_x(ux)
    ax = fd4_x(stress_xx) / RHO
    vx += 0.5*DT*ax
    ux += DT*vx
    stress_xx2 = (lam+2*mu)*fd4_x(ux)
    ax2 = fd4_x(stress_xx2) / RHO
    vx += 0.5*DT*ax2
    vx *= np.exp(-sigma_pml*DT)

    near_sig[step] = vx[SX_NEAR]
    far_sig[step] = vx[SX_FAR]

print('step | near_sig  | far_sig   | src_signal')
for i in range(NT):
    if abs(near_sig[i])>0.001 or abs(far_sig[i])>0.001 or abs(signal[i])>0.01:
        print(f'{i:4d} | {near_sig[i]:9.5f} | {far_sig[i]:9.5f} | {signal[i]:9.5f}')

print()
print(f'max|near|={np.max(np.abs(near_sig)):.5f} at step {np.argmax(np.abs(near_sig))}')
print(f'max|far| ={np.max(np.abs(far_sig)):.5f} at step {np.argmax(np.abs(far_sig))}')
print()
# Cross-correlation delay
from scipy.signal import correlate
corr = correlate(near_sig, far_sig, mode='full')
lags = np.arange(-(NT-1), NT)
delay_xcorr = lags[np.argmax(corr)]
print(f'Cross-correlation peak lag: {delay_xcorr} samples (expected 16.67)')
