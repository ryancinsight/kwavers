#!/usr/bin/env python3
"""
ivp_step_by_step_debug.py
========================
Compare pykwavers and pure Python PSTD field-by-field at every time step
to pinpoint exactly where (and how large) the divergence first appears.

Strategy:
- Small 16x16x16 grid (fast)
- Gaussian p0 varying only in x (1D-like, easy to visualise)
- ALL sensors (full 3D field at every step)
- Run pykwavers for N steps; reshape each column from Fortran order
- Run pure Python for same N steps
- Print Pearson r and max |diff| at each step

If the fields agree at step 1 but diverge later, the bug is in the time loop.
If they disagree at step 1 already, the bug is in initialisation or the first update.
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

# ── Grid / medium ──────────────────────────────────────────────────────────────
NX = NY = NZ = 16
DX = 1e-3 / 64   # same as larger tests: 15.625e-6 m
C0  = 1500.0
RHO0 = 1000.0
DT  = 2e-9
N_STEPS = 10
PML = 0

def pearson_r(a, b):
    af = a.flatten(); bf = b.flatten()
    am = af - af.mean(); bm = bf - bf.mean()
    d = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d) if d > 1e-30 else 0.0

# ── Build initial condition: 1D Gaussian in x ─────────────────────────────────
cx, cy, cz = NX//2, NY//2, NZ//2
sigma = 2.5
x_arr = np.arange(NX)
p0 = np.exp(-((x_arr - cx)**2) / (2*sigma**2))[:, None, None] * np.ones((NX, NY, NZ))
print(f"Grid: {NX}x{NY}x{NZ}  dx={DX*1e6:.3f} um  dt={DT*1e9:.1f} ns  c0={C0} m/s")
print(f"p0: Gaussian sigma={sigma} in x, peak={p0.max():.4f}, center=({cx},{cy},{cz})")

# ── Run pykwavers with ALL sensors ────────────────────────────────────────────
print(f"\nRunning pykwavers {N_STEPS} steps (ALL sensors)...")
grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
source = pkw.Source.from_initial_pressure(p0.copy())

sensor_mask = np.ones((NX, NY, NZ), dtype=bool)
sensor = pkw.Sensor.from_mask(sensor_mask)

sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
sim.set_pml_size(PML)

result = sim.run(time_steps=N_STEPS, dt=DT)
pkw_data = np.asarray(result.sensor_data, dtype=np.float64)
# pkw_data shape: (NX*NY*NZ, N_STEPS)
print(f"pykwavers sensor_data shape: {pkw_data.shape}")

# Reshape each time column: Fortran order (x fastest in SensorRecorder)
def pkw_field_at(t):
    """Return 3D field at step t (1-indexed: t=1 is after first update)."""
    return pkw_data[:, t-1].reshape(NX, NY, NZ, order='F')

# ── Run pure Python PSTD ──────────────────────────────────────────────────────
print(f"Running pure Python PSTD {N_STEPS} steps...")

kx_1d = np.fft.fftfreq(NX, d=DX) * 2 * np.pi
ky_1d = np.fft.fftfreq(NY, d=DX) * 2 * np.pi
kz_1d = np.fft.fftfreq(NZ, d=DX) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
KMAG = np.sqrt(KX**2 + KY**2 + KZ**2)

sinc_u = lambda x: np.where(np.abs(x) < 1e-12, 1.0, np.sin(x)/x)
kappa = sinc_u(C0 * DT * KMAG / 2.0)

ddx_pos = 1j * kx_1d * np.exp(+1j * kx_1d * DX / 2)
ddy_pos = 1j * ky_1d * np.exp(+1j * ky_1d * DX / 2)
ddz_pos = 1j * kz_1d * np.exp(+1j * kz_1d * DX / 2)
ddx_neg = 1j * kx_1d * np.exp(-1j * kx_1d * DX / 2)
ddy_neg = 1j * ky_1d * np.exp(-1j * ky_1d * DX / 2)
ddz_neg = 1j * kz_1d * np.exp(-1j * kz_1d * DX / 2)

DDX_P = ddx_pos[:, None, None]; DDX_N = ddx_neg[:, None, None]
DDY_P = ddy_pos[None, :, None]; DDY_N = ddy_neg[None, :, None]
DDZ_P = ddz_pos[None, None, :]; DDZ_N = ddz_neg[None, None, :]

p    = p0.astype(np.float64).copy()
ux   = np.zeros((NX, NY, NZ))
uy   = np.zeros((NX, NY, NZ))
uz   = np.zeros((NX, NY, NZ))
rhox = p / (3.0 * C0**2)
rhoy = p / (3.0 * C0**2)
rhoz = p / (3.0 * C0**2)

py_fields = []  # list of 3D arrays at steps 1..N_STEPS

for n in range(N_STEPS):
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
    p = C0**2 * (rhox + rhoy + rhoz)
    py_fields.append(p.copy())

print(f"Pure Python done.\n")

# ── Step-by-step comparison ───────────────────────────────────────────────────
print("=" * 80)
print(f"{'Step':>5}  {'r(pkw,py)':>10}  {'max|diff|':>12}  {'rms_py':>10}  {'rms_pkw':>10}  {'rms_ratio':>10}")
print("=" * 80)

for t in range(1, N_STEPS+1):
    p_pkw = pkw_field_at(t)
    p_py  = py_fields[t-1]
    r     = pearson_r(p_pkw, p_py)
    maxd  = float(np.abs(p_pkw - p_py).max())
    rms_py  = float(np.sqrt(np.mean(p_py**2)))
    rms_pkw = float(np.sqrt(np.mean(p_pkw**2)))
    ratio = rms_pkw / rms_py if rms_py > 1e-30 else float('nan')
    print(f"{t:5d}  {r:10.6f}  {maxd:12.4e}  {rms_py:10.4e}  {rms_pkw:10.4e}  {ratio:10.4f}")

# ── Detailed comparison at step 1 ─────────────────────────────────────────────
print("\n=== Detailed comparison at STEP 1 ===")
p_pkw1 = pkw_field_at(1)
p_py1  = py_fields[0]

print(f"\np0 stats:    max={p0.max():.6e}  min={p0.min():.6e}  rms={np.sqrt(np.mean(p0**2)):.6e}")
print(f"py  step1:   max={p_py1.max():.6e}  min={p_py1.min():.6e}  rms={np.sqrt(np.mean(p_py1**2)):.6e}")
print(f"pkw step1:   max={p_pkw1.max():.6e}  min={p_pkw1.min():.6e}  rms={np.sqrt(np.mean(p_pkw1**2)):.6e}")

# Along x at (cy, cz) — the axis of propagation
print(f"\nPressure profile along x at y={cy}, z={cz} after 1 step:")
print(f"{'ix':>4}  {'p0':>12}  {'py':>12}  {'pkw':>12}  {'diff':>12}")
for ix in range(NX):
    v0  = p0[ix, cy, cz]
    vpy = p_py1[ix, cy, cz]
    vpk = p_pkw1[ix, cy, cz]
    print(f"{ix:4d}  {v0:12.4e}  {vpy:12.4e}  {vpk:12.4e}  {vpk-vpy:12.4e}")

# ── Check Fortran vs C ordering at step 1 ────────────────────────────────────
print("\n=== Reshape order test at step 1 ===")
col0 = pkw_data[:, 0]  # first time step
p_F = col0.reshape(NX, NY, NZ, order='F')
p_C = col0.reshape(NX, NY, NZ, order='C')
r_F = pearson_r(p_F, p_py1)
r_C = pearson_r(p_C, p_py1)
print(f"Fortran reshape r vs py: {r_F:.6f}")
print(f"C       reshape r vs py: {r_C:.6f}")
print(f"(Higher is correct reshape order)")

# ── k-space amplitude check at step 1 ─────────────────────────────────────────
print("\n=== K-space amplitude ratio at step 1 ===")
p0_k    = np.fft.fftn(p0)
py1_k   = np.fft.fftn(p_py1)
pkw1_k  = np.fft.fftn(p_pkw1)

# Expected: py1_k / p0_k = kappa*(1 - kappa*c0²*dt²*|k|²) [leapfrog prediction]
theta = C0 * DT * KMAG / 2.0
kap   = sinc_u(theta)
leapfrog_ratio = kap * (1.0 - kap * 4.0 * theta**2)

# Compute actual ratios
mask_large = np.abs(p0_k) > 0.01 * np.abs(p0_k).max()
actual_py_ratio  = np.where(mask_large, np.abs(py1_k)  / (np.abs(p0_k)+1e-30), np.nan)
actual_pkw_ratio = np.where(mask_large, np.abs(pkw1_k) / (np.abs(p0_k)+1e-30), np.nan)
pred_ratio       = np.where(mask_large, leapfrog_ratio, np.nan)

py_err  = np.nanmax(np.abs(actual_py_ratio  - pred_ratio))
pkw_err = np.nanmax(np.abs(actual_pkw_ratio - pred_ratio))
print(f"Max |py_ratio  - leapfrog_ratio|: {py_err:.4e}  (should be ~0)")
print(f"Max |pkw_ratio - leapfrog_ratio|: {pkw_err:.4e}  (should be ~0 if correct)")

# Amplitude ratio pkw/py in k-space (should be ~1.0 everywhere)
amp_ratio_kspace = np.where(mask_large & (np.abs(py1_k) > 1e-30),
                            np.abs(pkw1_k) / np.abs(py1_k), np.nan)
print(f"K-space |pkw|/|py| at step 1:  mean={np.nanmean(amp_ratio_kspace):.6f}  "
      f"std={np.nanstd(amp_ratio_kspace):.6f}  "
      f"min={np.nanmin(amp_ratio_kspace):.4f}  max={np.nanmax(amp_ratio_kspace):.4f}")

# ── Velocity field test: does pykwavers use the same dt? ─────────────────────
# If dt_actual != dt_passed, velocity after step 1 would be scaled.
# We can back-calculate the effective dt from the step-1 pressure change.
# At DC (k=0): p_dc[1] = p0_dc (since sinc(0)=1, the DC component doesn't change)
# At k_nyquist: p[1] = p0 * kappa*(1 - kappa*c²dt²*k²)
# The sensitive check: find a non-DC k mode and compute what dt was used.
print("\n=== Back-calculate effective dt from step-1 field ===")
# Use k=(pi/dx, 0, 0): Nyquist in x, index NX//2
kx_nyq = np.pi / DX
kap_nyq = np.sinc(C0 * DT * kx_nyq / np.pi / 2)  # numpy sinc is normalized
# Wait — use our own sinc
arg_nyq = C0 * DT * kx_nyq / 2.0
kap_nyq2 = np.sin(arg_nyq)/arg_nyq if abs(arg_nyq) > 1e-12 else 1.0
print(f"Nyquist kx = {kx_nyq:.2f} rad/m")
print(f"Expected kappa(kx_nyq) = {kap_nyq2:.6f}")
print(f"Expected p[1]/p0 at Nyquist = {kap_nyq2*(1-kap_nyq2*4*arg_nyq**2):.6f}")

# 1D FFT along x at (cy, cz)
p0_x  = p0[:, cy, cz]
py1_x = p_py1[:, cy, cz]
pkw1_x = p_pkw1[:, cy, cz]
P0_x  = np.fft.fft(p0_x)
Py1_x = np.fft.fft(py1_x)
Pkw1_x = np.fft.fft(pkw1_x)

# At index 1 (lowest non-DC mode in x)
k1 = 2*np.pi / (NX*DX)
arg1 = C0 * DT * k1 / 2.0
kap1 = np.sin(arg1)/arg1
pred1 = kap1*(1 - kap1*4*arg1**2)
ratio_py1  = abs(Py1_x[1])  / (abs(P0_x[1])+1e-30)
ratio_pkw1 = abs(Pkw1_x[1]) / (abs(P0_x[1])+1e-30)
print(f"\nMode k1={k1:.2f} rad/m:")
print(f"  Expected p[1]/p0 ratio: {pred1:.8f}")
print(f"  py  actual ratio:       {ratio_py1:.8f}")
print(f"  pkw actual ratio:       {ratio_pkw1:.8f}")

# Back-calculate effective c0*dt from pkw ratio
# pkw_ratio = kappa(c*dt_eff*k/2) * (1 - kappa*c²*dt_eff²*k²)
# Solve numerically
from scipy.optimize import brentq
def f_ratio(cdt_eff, k, target_ratio):
    arg = cdt_eff * k / 2.0
    kap = np.sin(arg)/arg if abs(arg) > 1e-12 else 1.0
    return kap*(1 - kap*4*arg**2) - target_ratio

c0dt_nominal = C0 * DT  # = 3e-6 m
try:
    cdt_eff = brentq(f_ratio, 0.1*c0dt_nominal, 10*c0dt_nominal,
                     args=(k1, ratio_pkw1), xtol=1e-15)
    c_eff = cdt_eff / DT
    print(f"\nBack-calculated c_eff from pkw step-1 k1 ratio: {c_eff:.2f} m/s")
    print(f"  (nominal c0={C0:.2f} m/s, ratio c_eff/c0={c_eff/C0:.4f})")
except Exception as e:
    print(f"\nBack-calc failed: {e}")
    print(f"  pkw ratio={ratio_pkw1:.8f}; brent search failed — checking range")
    for f in [0.5, 1.0, 1.1, 1.2, 1.5, 2.0]:
        cdt_t = f * c0dt_nominal
        k = k1; arg = cdt_t*k/2; kap2 = np.sin(arg)/arg if abs(arg)>1e-12 else 1.0
        r_t = kap2*(1-kap2*4*arg**2)
        print(f"  c_eff={f*C0:.1f}: ratio={r_t:.8f}")

print("\nDone.")
