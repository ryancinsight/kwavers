#!/usr/bin/env python3
"""
ivp_dt_scaling_test.py
======================
Empirically determine the effective wave speed in pykwavers by testing
how the step-1 pressure change scales with dt.

For the PSTD leapfrog, at small dt:
  p[1]_k / p0_k ≈ 1 - c_eff²*dt²*k²  (leading-order correction)

So (1 - ratio) / dt² / k² = c_eff². Sweeping dt gives us c_eff.

Also: compare with and without source_kappa to isolate initialization vs propagation.
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

NX = NY = NZ = 16
DX = 1e-3 / 64   # 15.625e-6 m  (same as larger tests)
C0  = 1500.0
RHO0 = 1000.0
PML = 0

def sinc_u(x):
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x)/x)

cx, cy, cz = NX//2, NY//2, NZ//2
sigma = 2.5
x_arr = np.arange(NX)
p0 = np.exp(-((x_arr - cx)**2) / (2*sigma**2))[:, None, None] * np.ones((NX, NY, NZ))

# 1D k-vector for x
dk = 2*np.pi / (NX * DX)
kx_vec = np.array([i*dk if i <= NX//2 else (i-NX)*dk for i in range(NX)])

print(f"Grid: {NX}x{NX}x{NX}  DX={DX*1e6:.3f} um  C0={C0}  RHO0={RHO0}")
print(f"k1 = dk = {dk:.2f} rad/m  (lowest non-DC mode)")
print()

def run_1step(dt):
    """Run pykwavers for 1 step, return 3D field (Fortran reshape)."""
    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_initial_pressure(p0.copy())
    sensor_mask = np.ones((NX, NY, NZ), dtype=bool)
    sensor = pkw.Sensor.from_mask(sensor_mask)
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML)
    result = sim.run(time_steps=1, dt=dt)
    data = np.asarray(result.sensor_data, dtype=np.float64)
    return data[:, 0].reshape(NX, NY, NZ, order='F')

# ── DT sweep: measure (1 - ratio) / (dt²*k1²) ────────────────────────────────
print("=== DT scaling test: back-calculate c_eff from step-1 k-space ratio ===")
print(f"{'DT (ns)':>10}  {'ratio_k1':>10}  {'1-ratio':>10}  {'c_eff (m/s)':>12}  {'c_eff/c0':>10}")
print("-" * 60)

# Also compute pure Python reference at each dt
for dt_ns in [0.1, 0.5, 1.0, 2.0, 5.0]:
    dt = dt_ns * 1e-9

    p1_pkw = run_1step(dt)
    p1_x   = p1_pkw[:, cy, cz]
    p0_x   = p0[:, cy, cz]

    P1  = np.fft.fft(p1_x)
    P0  = np.fft.fft(p0_x)

    # At mode index 1 (k = dk = 25133 rad/m for NX=16)
    ratio = abs(P1[1]) / (abs(P0[1]) + 1e-30)
    one_minus_ratio = 1 - ratio
    k1 = dk

    # c_eff² = (1 - ratio) / (dt² * k1²)  approximately
    # but this includes source_kappa contribution too
    c_eff_sq = one_minus_ratio / (dt**2 * k1**2) if dt > 0 else float('nan')
    c_eff = np.sqrt(c_eff_sq) if c_eff_sq > 0 else float('nan')

    print(f"{dt_ns:10.1f}  {ratio:10.6f}  {one_minus_ratio:10.6f}  {c_eff:12.2f}  {c_eff/C0:10.4f}")

# ── Precise comparison at DT=2e-9: py vs pkw in 3D k-space ─────────────────
print()
print("=== Step-1 k-space ratio (pkw vs py) across modes, DT=2e-9 ===")
DT = 2e-9

# Run pkw
p1_pkw = run_1step(DT)

# Run pure Python
kx_1d = np.fft.fftfreq(NX, d=DX) * 2 * np.pi
ky_1d = np.fft.fftfreq(NY, d=DX) * 2 * np.pi
kz_1d = np.fft.fftfreq(NZ, d=DX) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
KMAG = np.sqrt(KX**2 + KY**2 + KZ**2)
kappa = sinc_u(C0 * DT * KMAG / 2.0)

DDX_P = (1j * kx_1d * np.exp(+1j * kx_1d * DX / 2))[:, None, None]
DDY_P = (1j * ky_1d * np.exp(+1j * ky_1d * DX / 2))[None, :, None]
DDZ_P = (1j * kz_1d * np.exp(+1j * kz_1d * DX / 2))[None, None, :]
DDX_N = (1j * kx_1d * np.exp(-1j * kx_1d * DX / 2))[:, None, None]
DDY_N = (1j * ky_1d * np.exp(-1j * ky_1d * DX / 2))[None, :, None]
DDZ_N = (1j * kz_1d * np.exp(-1j * kz_1d * DX / 2))[None, None, :]

p    = p0.astype(np.float64).copy()
ux   = np.zeros((NX, NY, NZ))
uy   = np.zeros((NX, NY, NZ))
uz   = np.zeros((NX, NY, NZ))
rhox = p / (3.0 * C0**2)
rhoy = p / (3.0 * C0**2)
rhoz = p / (3.0 * C0**2)

pk = np.fft.fftn(p)
ux -= DT / RHO0 * np.real(np.fft.ifftn(kappa * DDX_P * pk))
uy -= DT / RHO0 * np.real(np.fft.ifftn(kappa * DDY_P * pk))
uz -= DT / RHO0 * np.real(np.fft.ifftn(kappa * DDZ_P * pk))
uxk = np.fft.fftn(ux)
uyk = np.fft.fftn(uy)
uzk = np.fft.fftn(uz)
rhox -= DT * RHO0 * np.real(np.fft.ifftn(kappa * DDX_N * uxk))
rhoy -= DT * RHO0 * np.real(np.fft.ifftn(kappa * DDY_N * uyk))
rhoz -= DT * RHO0 * np.real(np.fft.ifftn(kappa * DDZ_N * uzk))
p1_py = C0**2 * (rhox + rhoy + rhoz)

# 3D k-space of p0, p1_py, p1_pkw
P0_k   = np.fft.fftn(p0)
P1_py_k  = np.fft.fftn(p1_py)
P1_pkw_k = np.fft.fftn(p1_pkw)

# Theoretical prediction
theta = C0 * DT * KMAG / 2.0
leapfrog_ratio = sinc_u(theta) * (1.0 - 4.0 * sinc_u(theta)**2 * theta**2)
with_sk_ratio = np.cos(theta) * (1.0 - 4.0 * sinc_u(theta)**2 * theta**2)

mask = np.abs(P0_k) > 0.01 * np.abs(P0_k).max()

actual_py_r  = np.where(mask, np.abs(P1_py_k)  / np.abs(P0_k), np.nan)
actual_pkw_r = np.where(mask, np.abs(P1_pkw_k) / np.abs(P0_k), np.nan)
pred_no_sk   = np.where(mask, leapfrog_ratio,  np.nan)
pred_with_sk = np.where(mask, with_sk_ratio,   np.nan)

print(f"{'':20}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
print(f"{'py ratio/p0':20}  {np.nanmean(actual_py_r):10.6f}  {np.nanstd(actual_py_r):10.6f}  {np.nanmin(actual_py_r):10.6f}  {np.nanmax(actual_py_r):10.6f}")
print(f"{'pkw ratio/p0':20}  {np.nanmean(actual_pkw_r):10.6f}  {np.nanstd(actual_pkw_r):10.6f}  {np.nanmin(actual_pkw_r):10.6f}  {np.nanmax(actual_pkw_r):10.6f}")
print(f"{'pred_no_sk':20}  {np.nanmean(pred_no_sk):10.6f}  {np.nanstd(pred_no_sk):10.6f}  {np.nanmin(pred_no_sk):10.6f}  {np.nanmax(pred_no_sk):10.6f}")
print(f"{'pred_with_sk':20}  {np.nanmean(pred_with_sk):10.6f}  {np.nanstd(pred_with_sk):10.6f}  {np.nanmin(pred_with_sk):10.6f}  {np.nanmax(pred_with_sk):10.6f}")

print()
# Check the k-space ratio pkw/py per k-shell
print("=== k-space ratio pkw/py vs KMAG (by shell) ===")
print(f"{'k_mag (rad/m)':>15}  {'n_modes':>8}  {'ratio_pkw/py':>14}  {'pred_cos':>12}")
k_mag_vals = KMAG[mask]
ratio_pkw_py = np.abs(P1_pkw_k[mask]) / (np.abs(P1_py_k[mask]) + 1e-30)
cos_pred = np.cos(C0 * DT * k_mag_vals / 2.0)

# Bin by k_mag
bins = np.linspace(0, k_mag_vals.max()*1.01, 8)
for b_lo, b_hi in zip(bins[:-1], bins[1:]):
    sel = (k_mag_vals >= b_lo) & (k_mag_vals < b_hi)
    if sel.sum() == 0:
        continue
    r_mean = ratio_pkw_py[sel].mean()
    c_mean = cos_pred[sel].mean()
    k_mean = k_mag_vals[sel].mean()
    print(f"{k_mean:15.0f}  {sel.sum():8d}  {r_mean:14.6f}  {c_mean:12.6f}")

print()
print("If ratio_pkw/py ≈ cos(c0*dt*k/2): source_kappa is correct, propagation is same")
print("If ratio_pkw/py > cos: pkw propagates extra energy (c_eff > c0 in kappa?)")
print("If ratio_pkw/py < cos: source_kappa too large (c_ref_sk > c0?)")

# ── Check if pure Python matches theoretical ratio ────────────────────────────
print()
print("=== Sanity check: py vs theoretical (no source_kappa) ===")
py_vs_theory = np.where(mask, np.abs(P1_py_k) / (np.abs(P0_k)*np.abs(leapfrog_ratio)+1e-30), np.nan)
print(f"py/leapfrog_pred: mean={np.nanmean(py_vs_theory):.6f}  std={np.nanstd(py_vs_theory):.6f}")

pkw_vs_theory_sk = np.where(mask & (np.abs(with_sk_ratio) > 0.01),
                             np.abs(P1_pkw_k) / (np.abs(P0_k)*np.abs(with_sk_ratio)+1e-30), np.nan)
print(f"pkw/with_sk_pred: mean={np.nanmean(pkw_vs_theory_sk):.6f}  std={np.nanstd(pkw_vs_theory_sk):.6f}")
print()
print("If pkw/with_sk_pred ≈ 1.0: pykwavers exactly implements source_kappa + correct leapfrog")
print("If pkw/with_sk_pred ≠ 1.0: there's an additional discrepancy")
