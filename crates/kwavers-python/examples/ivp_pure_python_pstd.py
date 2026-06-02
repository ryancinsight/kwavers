#!/usr/bin/env python3
"""
ivp_pure_python_pstd.py
=======================
Pure-Python PSTD reference implementation of k-Wave's update equations,
run side-by-side against the cached k-Wave and pykwavers results.

Goal: determine whether the 7-step timing offset is a pykwavers bug
      or whether k-Wave does something extra for IVP initialization.

This script implements the k-Wave MATLAB kspaceFirstOrder3D equations
verbatim:
  1. kappa = ifftshift(sinc_unnorm(c_ref * |k| * dt / 2))
  2. ddx_k_shift_pos[i] = i*kx[i] * exp(+i*kx[i]*dx/2)  (FFT-order k)
  3. ddx_k_shift_neg[i] = i*kx[i] * exp(-i*kx[i]*dx/2)
  4. For each step:
       ux = ux - dt/rho0 * real(ifftn(kappa * ddx_pos * fftn(p)))
       rho_x = rho_x - dt*rho0 * real(ifftn(kappa * ddx_neg * fftn(ux)))
       (same for y, z)
       p = c0^2 * (rho_x + rho_y + rho_z)
       record p at sensor
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from pathlib import Path
from scipy import signal as sp_signal

OUTPUT_DIR = Path(__file__).parent / "output"
KWAVE_CACHE  = OUTPUT_DIR / "ivp_photoacoustic_kwave_cache.npz"
PKWAV_CACHE  = OUTPUT_DIR / "ivp_photoacoustic_pykwavers_cache.npz"

# ── Grid / medium constants (must match compare script exactly) ──────────────
NX = NY = NZ = 64
DX = 1e-3 / NX   # 15.625e-6 m
C0 = 1500.0
RHO0 = 1000.0
DT = 2e-9
NT = 150

SOURCE_RADIUS = 2
SENSOR_IX, SENSOR_IY, SENSOR_IZ = 42, 32, 32
BALL_CENTER_0IDX = (31, 31, 31)  # 0-indexed

def sinc_unnorm(x):
    """sin(x)/x, unnormalized sinc."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / x)

def make_ball(center, radius, shape):
    cx, cy, cz = center
    x = np.arange(shape[0]); y = np.arange(shape[1]); z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) <= radius**2

def kwave_smooth_1d(arr_k, restore_max):
    """Blackman-window smooth in spectral domain (k-Wave's smooth() function)."""
    # k-Wave uses a 3D Blackman window in k-space, applied as multiply by window
    # The simplest implementation: not needed here since we load cached p0_smooth
    pass

def pearson_r(a, b):
    am = a - a.mean(); bm = b - b.mean()
    d = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d) if d > 1e-30 else 0.0

def run_pure_python_pstd(p0):
    """
    Pure Python PSTD matching k-Wave MATLAB kspaceFirstOrder3D.

    Uses:
    - kappa = ifftshift(sinc_unnorm(c_ref * |k| * dt / 2))  [in FFT output order]
    - shift_pos[axis][i] = i*k[i] * exp(+i*k[i]*ds/2)
    - shift_neg[axis][i] = i*k[i] * exp(-i*k[i]*ds/2)
    - No PML (clean periodic BC test; reflections arrive after step 200+)
    - IVP: p[0] = p0, rho = p0/(3*c0^2), u = 0
    """
    shape = (NX, NY, NZ)

    # ── k-vectors (MATLAB/numpy FFT order, NOT shifted) ──────────────────────
    # k-Wave MATLAB: kx_vec = (2*pi/Nx/dx) * [0,1,...,Nx/2-1,-Nx/2,...,-1]
    kx_1d = np.fft.fftfreq(NX, d=DX) * 2 * np.pi   # [0,1,...,-1]*2pi/(Nx*dx)
    ky_1d = np.fft.fftfreq(NY, d=DX) * 2 * np.pi
    kz_1d = np.fft.fftfreq(NZ, d=DX) * 2 * np.pi

    # For kappa, k-Wave uses |k| in "centered" coordinates then ifftshift.
    # In MATLAB: k = ifftshift(sqrt(kx.^2 + ky.^2 + kz.^2)) where kx,ky,kz are
    # from centered meshgrid. But actually in MATLAB the k vectors are already
    # in fftshift order in the kgrid object:
    #   kgrid.kx = fftshift((2*pi/Nx/dx) * [-Nx/2,...,Nx/2-1])
    # And kappa = ifftshift(sinc_unnorm(c_ref * kgrid.k * dt / 2))
    # So the final kappa IS in FFT output order (ready to multiply with fftn output).
    #
    # Let's compute directly in FFT order:
    KX, KY, KZ = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    KMAG = np.sqrt(KX**2 + KY**2 + KZ**2)   # in FFT output order
    kappa = sinc_unnorm(C0 * DT * KMAG / 2.0)  # in FFT output order (NO ifftshift needed!)
    # Note: ifftshift(sinc_unnorm(c_ref * |k_centered| * dt / 2)) with k_centered
    #       in shifted order = sinc_unnorm(c_ref * |k_fft| * dt / 2) in FFT order.
    #       They are equivalent since sinc_unnorm is symmetric and only depends on |k|.

    # ── Staggered-grid shift operators ───────────────────────────────────────
    # shift_pos for pressure → velocity gradient: ik * exp(+ik*ds/2)
    # shift_neg for velocity → density divergence: ik * exp(-ik*ds/2)
    def make_shift_ops(k1d, ds):
        shift_pos = 1j * k1d * np.exp(+1j * k1d * ds / 2)  # (N,)
        shift_neg = 1j * k1d * np.exp(-1j * k1d * ds / 2)  # (N,)
        return shift_pos, shift_neg

    ddx_pos, ddx_neg = make_shift_ops(kx_1d, DX)
    ddy_pos, ddy_neg = make_shift_ops(ky_1d, DX)
    ddz_pos, ddz_neg = make_shift_ops(kz_1d, DX)

    # Broadcast to 3D arrays (for element-wise multiplication in k-space)
    DDX_POS = ddx_pos[:, None, None] * np.ones((NX, NY, NZ))
    DDX_NEG = ddx_neg[:, None, None] * np.ones((NX, NY, NZ))
    DDY_POS = ddy_pos[None, :, None] * np.ones((NX, NY, NZ))
    DDY_NEG = ddy_neg[None, :, None] * np.ones((NX, NY, NZ))
    DDZ_POS = ddz_pos[None, None, :] * np.ones((NX, NY, NZ))
    DDZ_NEG = ddz_neg[None, None, :] * np.ones((NX, NY, NZ))

    # ── Initial conditions ────────────────────────────────────────────────────
    p   = p0.astype(np.float64).copy()
    ux  = np.zeros(shape)
    uy  = np.zeros(shape)
    uz  = np.zeros(shape)
    rhox = p / (3.0 * C0**2)
    rhoy = p / (3.0 * C0**2)
    rhoz = p / (3.0 * C0**2)

    # Helper: spectral gradient
    def spectral_grad(field, shift_3d):
        """real(ifftn(shift_3d * fftn(field)))"""
        return np.real(np.fft.ifftn(shift_3d * np.fft.fftn(field)))

    # ── Time loop ─────────────────────────────────────────────────────────────
    sensor_trace = np.zeros(NT)

    for n in range(NT):
        # Step 1: Velocity update
        pk = np.fft.fftn(p)
        ux  -= DT / RHO0 * np.real(np.fft.ifftn(kappa * DDX_POS * pk))
        uy  -= DT / RHO0 * np.real(np.fft.ifftn(kappa * DDY_POS * pk))
        uz  -= DT / RHO0 * np.real(np.fft.ifftn(kappa * DDZ_POS * pk))

        # Step 2: Density update
        uxk = np.fft.fftn(ux)
        uyk = np.fft.fftn(uy)
        uzk = np.fft.fftn(uz)
        rhox -= DT * RHO0 * np.real(np.fft.ifftn(kappa * DDX_NEG * uxk))
        rhoy -= DT * RHO0 * np.real(np.fft.ifftn(kappa * DDY_NEG * uyk))
        rhoz -= DT * RHO0 * np.real(np.fft.ifftn(kappa * DDZ_NEG * uzk))

        # Step 3: Pressure update
        p = C0**2 * (rhox + rhoy + rhoz)

        # Record sensor
        sensor_trace[n] = p[SENSOR_IX, SENSOR_IY, SENSOR_IZ]

    return sensor_trace

def first_sign_change_after_peak(arr):
    pk = int(np.argmax(np.abs(arr)))
    for i in range(pk, len(arr)-1):
        if arr[i] * arr[i+1] < 0:
            return i
    return -1

def main():
    # Load p0_smooth from the k-Wave Python cache (it IS the smoothed p0)
    # Actually we need to rebuild p0 from scratch and smooth it
    # to ensure we're using the same initial condition as both engines.
    try:
        from kwave.utils.mapgen import make_ball as kwave_make_ball
        from kwave.data import Vector
        from kwave.utils.filters import smooth as kwave_smooth
        ball_mask = kwave_make_ball(
            Vector([NX, NY, NZ]),
            Vector([NX//2, NY//2, NZ//2]),  # 1-indexed center: [32,32,32]
            SOURCE_RADIUS
        )
        p0_raw = np.asarray(ball_mask, dtype=np.float64)
        p0_smooth = kwave_smooth(p0_raw, restore_max=True)
        print(f"p0 built: {int(p0_raw.sum())} pts in ball, smooth peak={p0_smooth.max():.6f}")
    except ImportError:
        print("kwave not available — using raw ball (no smoothing)")
        p0_smooth = np.zeros((NX, NY, NZ))
        cx, cy, cz = BALL_CENTER_0IDX
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    if (i-cx)**2+(j-cy)**2+(k-cz)**2 <= SOURCE_RADIUS**2:
                        p0_smooth[i,j,k] = 1.0

    # Load cached results
    have_cache = KWAVE_CACHE.exists() and PKWAV_CACHE.exists()
    if have_cache:
        kw  = np.load(KWAVE_CACHE)["trace"].flatten().astype(np.float64)
        pkw = np.load(PKWAV_CACHE)["trace"].flatten().astype(np.float64)
        print(f"Loaded k-Wave cache ({len(kw)} pts) and pykwavers cache ({len(pkw)} pts)")
    else:
        print("Cache not found — run ivp_photoacoustic_waveforms_compare.py first")
        kw  = None
        pkw = None

    # Run pure Python PSTD
    print("\nRunning pure Python PSTD...")
    py_trace = run_pure_python_pstd(p0_smooth)
    print(f"Pure Python PSTD done. peak={np.abs(py_trace).max():.6e} Pa  rms={np.sqrt(np.mean(py_trace**2)):.6e} Pa")

    # Timing analysis
    t_ns = np.arange(NT) * DT * 1e9
    py_zc = first_sign_change_after_peak(py_trace)
    print(f"\nPure Python PSTD zero-cross: step {py_zc+1}  (t={t_ns[py_zc]:.1f} ns)")

    if have_cache:
        kw_zc  = first_sign_change_after_peak(kw)
        pkw_zc = first_sign_change_after_peak(pkw)
        print(f"k-Wave          zero-cross: step {kw_zc+1}  (t={t_ns[kw_zc]:.1f} ns)")
        print(f"pykwavers       zero-cross: step {pkw_zc+1}  (t={t_ns[pkw_zc]:.1f} ns)")

    phy_t_ns = np.sqrt((SENSOR_IX-BALL_CENTER_0IDX[0])**2 +
                       (SENSOR_IY-BALL_CENTER_0IDX[1])**2 +
                       (SENSOR_IZ-BALL_CENTER_0IDX[2])**2) * DX / C0 * 1e9
    phy_step = phy_t_ns / (DT * 1e9)
    print(f"Physical arrival: {phy_t_ns:.1f} ns = step {phy_step:.1f}")

    # Pearson r comparisons
    print("\nPearson r (no shift):")
    if have_cache:
        print(f"  py-PSTD vs k-Wave:    {pearson_r(py_trace, kw):.6f}")
        print(f"  py-PSTD vs pykwavers: {pearson_r(py_trace, pkw):.6f}")
        print(f"  k-Wave vs pykwavers:  {pearson_r(kw, pkw):.6f}")

    # Cross-correlation
    if have_cache:
        print("\nCross-correlation shifts:")
        for (name_a, a), (name_b, b) in [
            (("py-PSTD", py_trace), ("k-Wave", kw)),
            (("py-PSTD", py_trace), ("pykwavers", pkw)),
        ]:
            corr = sp_signal.correlate(a, b, mode="full")
            lags = sp_signal.correlation_lags(len(a), len(b), mode="full")
            best_lag = int(lags[np.argmax(corr)])
            print(f"  {name_a} vs {name_b}: best_lag={best_lag} (positive = {name_b} leads {name_a})")

    # Shift sweep: py-PSTD vs k-Wave
    if have_cache:
        print("\nShift sweep: py-PSTD vs k-Wave (positive shift delays py-PSTD):")
        for shift in range(-3, 12):
            if shift >= 0:
                py_s = np.concatenate([np.zeros(shift), py_trace[:-shift if shift else None]])
            else:
                py_s = np.concatenate([py_trace[-shift:], np.zeros(-shift)])
            r_s = pearson_r(kw, py_s)
            flag = " <-- zero" if shift == 0 else (" <-- best?" if abs(r_s) >= 0.97 else "")
            print(f"  shift={shift:+3d}: r={r_s:+.4f}{flag}")

    # Save pure Python trace for the timing diagnostic
    out_path = OUTPUT_DIR / "ivp_pure_python_pstd_cache.npz"
    np.savez(str(out_path), trace=py_trace, runtime_s=0.0)
    print(f"\nSaved pure Python trace to {out_path}")

    # Show a few steps around the wavefront for both
    print(f"\nTrace comparison around wavefront (steps 47-65):")
    print(f"{'Step':>5}  {'t_ns':>7}  {'py-PSTD':>12}  {'k-Wave':>12}  {'pykwavers':>12}")
    for i in range(46, min(66, NT)):
        kw_val  = kw[i]  if have_cache else float('nan')
        pkw_val = pkw[i] if have_cache else float('nan')
        print(f"{i+1:5d}  {t_ns[i]:7.1f}  {py_trace[i]:12.4e}  {kw_val:12.4e}  {pkw_val:12.4e}")

if __name__ == "__main__":
    main()
