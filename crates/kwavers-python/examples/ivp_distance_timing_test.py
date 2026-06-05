#!/usr/bin/env python3
"""
ivp_distance_timing_test.py
Test the IVP timing offset at MULTIPLE sensor distances to determine:
  - Fixed offset (same N steps regardless of distance) → initialization/recording bug
  - Proportional offset (N steps ∝ distance) → effective wave speed error

Also tests a 1D-like Gaussian (pure x-direction propagation) for simplicity.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[3]
_pyp  = _root / "crates" / "kwavers-python" / "python"
if str(_pyp) not in sys.path:
    sys.path.insert(0, str(_pyp))

import pykwavers as pkw
from kwave.utils.filters import smooth as kwave_smooth

NX = NY = NZ = 64
DX = 1e-3 / NX   # 15.625e-6 m
C0 = 1500.0
RHO0 = 1000.0
DT = 2e-9
NT = 150
PML = 0   # no PML to avoid reflections at large distances

def pearson_r(a, b):
    am = a - a.mean(); bm = b - b.mean()
    d = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d) if d > 1e-30 else 0.0

def run_pykwavers(p0, sensors_dict):
    """Run pykwavers IVP, sensors_dict = {name: (ix, iy, iz)}"""
    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_initial_pressure(p0.copy())

    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    for (ix, iy, iz) in sensors_dict.values():
        sensor_mask[ix, iy, iz] = True
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML)

    result = sim.run(time_steps=NT, dt=DT)
    # sensor_data shape: (n_sensors, NT)
    data = np.asarray(result.sensor_data, dtype=np.float64)

    # Map back to sensor names - sensor indices match the order of True entries in mask
    # (row-major order: ix varies slowest, iz fastest)
    true_indices = np.argwhere(sensor_mask)  # shape (n_sensors, 3), sorted by [ix,iy,iz]
    traces = {}
    for j, (name, pos) in enumerate(sensors_dict.items()):
        # Find which row of true_indices matches pos
        for k, idx in enumerate(true_indices):
            if list(idx) == list(pos):
                traces[name] = data[k] if data.ndim == 2 else data.flatten()
                break
    return traces

def run_pure_python(p0, sensors_dict):
    """Pure Python PSTD reference, sensors_dict = {name: (ix, iy, iz)}"""
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

    traces = {name: np.zeros(NT) for name in sensors_dict}

    for n in range(NT):
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
        for name, (ix, iy, iz) in sensors_dict.items():
            traces[name][n] = p[ix, iy, iz]

    return traces

def find_zero_crossing_physical(arr, threshold=1e-3):
    """Find first + → - zero-crossing where the signal is above threshold."""
    peak_val = np.abs(arr).max()
    for i in range(1, len(arr)-1):
        if arr[i] > 0 and arr[i+1] < 0 and arr[i] > threshold * peak_val:
            return i  # 0-indexed
    return -1

def main():
    cx, cy, cz = 31, 31, 31   # ball center (0-indexed)
    source_radius = 2

    print("=" * 70)
    print("IVP timing at multiple sensor distances: fixed or proportional?")
    print("=" * 70)

    # ── Test 1: Gaussian p0 (1D-like, varying in x only) ─────────────────────
    print("\n=== Test 1: 1D Gaussian p0 (sigma=3 in x, const in y,z) ===")
    sigma = 3.0
    x_arr = np.arange(NX)
    p0_1d = np.exp(-((x_arr - cx)**2) / (2*sigma**2))[:, None, None] * np.ones((NX, NY, NZ))
    # Sensor at different x positions (same y=31, z=31 as source center)
    sensors_1d = {
        "x=cx+6 (D=6)": (cx+6, cy, cz),
        "x=cx+10 (D=10)": (cx+10, cy, cz),
        "x=cx+16 (D=16)": (cx+16, cy, cz),
    }

    print("\nRunning pykwavers (1D Gaussian)...")
    tr_pkw_1d = run_pykwavers(p0_1d, sensors_1d)
    print("Running pure Python PSTD (1D Gaussian)...")
    tr_py_1d = run_pure_python(p0_1d, sensors_1d)

    t_ns = np.arange(NT) * DT * 1e9
    print(f"\n{'Sensor':>18}  {'D[gp]':>6}  {'expected_ns':>12}  {'py_zc_ns':>10}  {'pkw_zc_ns':>11}  {'offset_steps':>13}")
    for name, (ix, iy, iz) in sensors_1d.items():
        D = np.sqrt((ix-cx)**2 + (iy-cy)**2 + (iz-cz)**2)
        t_phys = D * DX / C0 * 1e9
        zc_py  = find_zero_crossing_physical(tr_py_1d[name])
        zc_pkw = find_zero_crossing_physical(tr_pkw_1d[name])
        t_py   = t_ns[zc_py]  if zc_py  >= 0 else float('nan')
        t_pkw  = t_ns[zc_pkw] if zc_pkw >= 0 else float('nan')
        offset = (zc_pkw - zc_py) if (zc_py >= 0 and zc_pkw >= 0) else float('nan')
        print(f"{name:>18}  {D:6.2f}  {t_phys:12.1f}  {t_py:10.1f}  {t_pkw:11.1f}  {offset:13.1f}")

    # ── Test 2: 3D ball source, multiple sensors ──────────────────────────────
    print("\n=== Test 2: 3D ball source, multiple sensor distances ===")
    from kwave.utils.mapgen import make_ball
    from kwave.data import Vector
    ball_mask = make_ball(Vector([NX,NY,NZ]), Vector([NX//2,NY//2,NZ//2]), source_radius)
    p0_ball   = np.asarray(ball_mask, dtype=np.float64)
    p0_smooth = kwave_smooth(p0_ball, restore_max=True)

    # Sensors at various distances along x axis (sensor at [cx+d, cy, cz])
    sensors_ball = {}
    for d in [4, 7, 10, 14, 18]:
        ix_s = cx + d
        dist = np.sqrt((ix_s-cx)**2 + (cy-cy)**2 + (cz-cz)**2)  # = d exactly
        if ix_s < NX:
            sensors_ball[f"x=cx+{d} (D={dist:.1f})"] = (ix_s, cy, cz)

    print(f"Running pykwavers (ball, no PML)...")
    tr_pkw_ball = run_pykwavers(p0_smooth, sensors_ball)
    print(f"Running pure Python PSTD (ball, no PML)...")
    tr_py_ball  = run_pure_python(p0_smooth, sensors_ball)

    print(f"\n{'Sensor':>25}  {'D[gp]':>6}  {'expected_ns':>12}  {'py_zc_ns':>10}  {'pkw_zc_ns':>11}  {'offset_steps':>13}")
    for name, (ix, iy, iz) in sensors_ball.items():
        D = np.sqrt((ix-cx)**2 + (iy-cy)**2 + (iz-cz)**2)
        t_phys = D * DX / C0 * 1e9
        zc_py  = find_zero_crossing_physical(tr_py_ball[name])
        zc_pkw = find_zero_crossing_physical(tr_pkw_ball[name])
        t_py   = t_ns[zc_py]  if zc_py  >= 0 else float('nan')
        t_pkw  = t_ns[zc_pkw] if zc_pkw >= 0 else float('nan')
        offset = (zc_pkw - zc_py) if (zc_py >= 0 and zc_pkw >= 0) else float('nan')
        print(f"{name:>25}  {D:6.2f}  {t_phys:12.1f}  {t_py:10.1f}  {t_pkw:11.1f}  {offset:13.1f}")

    # ── Report conclusions ─────────────────────────────────────────────────────
    print("\n=== Analysis ===")
    print("If 'offset_steps' is constant (same for all D): FIXED initialization/recording bug")
    print("If 'offset_steps' grows proportionally with D: WAVE SPEED ERROR (c_eff > c0)")

if __name__ == "__main__":
    main()
