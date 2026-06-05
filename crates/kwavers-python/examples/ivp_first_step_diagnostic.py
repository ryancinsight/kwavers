#!/usr/bin/env python3
"""
ivp_first_step_diagnostic.py
Test pykwavers' first PSTD step against the analytical PSTD prediction.

For an IVP with initial pressure p0, the EXACT spectral solution after 1 step:
  p[1]_k = p0_k * cos(c0 * dt * |k|)

For pykwavers' leapfrog with u[-1/2]=0 and rho[0]=p0/c0²:
  u[1/2]_k = -dt/rho0 * kappa * shift_pos * p0_k
  rho_x[1]_k = rho_x[0]_k - dt*rho0 * kappa * shift_neg * u_x[1/2]_k
             = p0/(3c0²) - dt²*kappa²*k_x²*p0
  Similarly for y and z.
  p[1]_k = c0² * (rho_x[1] + rho_y[1] + rho_z[1])
          = p0_k * (kappa - kappa²*c0²*dt²*|k|²)   [kappa = sinc(c0*dt*|k|/2)]

This deviates from cos(c0*dt*|k|) for high-k modes.

The diagnostic:
1. Creates simple initial condition (e.g. Gaussian p0)
2. Runs pykwavers for 1 step
3. Compares with analytical prediction
4. Plots the discrepancy in k-space
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from pathlib import Path

# ─── Try to bootstrap pykwavers ───────────────────────────────────────────────
_root = Path(__file__).resolve().parents[3]
_pyp  = _root / "crates" / "kwavers-python" / "python"
if str(_pyp) not in sys.path:
    sys.path.insert(0, str(_pyp))

import pykwavers as pkw

# ─── Grid / medium constants matching the IVP comparison ──────────────────────
NX = NY = NZ = 16      # small grid for speed
DX  = 15.625e-6        # [m]
C0  = 1500.0           # [m/s]
RHO0 = 1000.0          # [kg/m³]
DT  = 2e-9             # [s]
PML = 0                # no PML for clean test

def sinc_unnorm(x):
    """Unnormalized sinc: sin(x)/x"""
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(np.abs(x) < 1e-10, 1.0, np.sin(x) / x)
    return s

def main():
    print("=" * 60)
    print("IVP First-Step Diagnostic")
    print(f"  Grid   : {NX}x{NY}x{NZ}  dx={DX*1e6:.3f} um")
    print(f"  c0     : {C0} m/s  rho0={RHO0} kg/m3  dt={DT*1e9:.1f} ns")
    print("=" * 60)

    # ── Create Gaussian p0 ────────────────────────────────────────────────────
    cx, cy, cz = NX//2, NY//2, NZ//2
    sigma = 2.0   # grid points
    x = np.arange(NX); y = np.arange(NY); z = np.arange(NZ)
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
    r2 = (XX - cx)**2 + (YY - cy)**2 + (ZZ - cz)**2
    p0 = np.exp(-r2 / (2 * sigma**2))   # Gaussian, peak=1.0

    # ── Analytical prediction for p[1] ────────────────────────────────────────
    # k vectors in FFT order
    dk_x = 2*np.pi / (NX * DX)
    dk_y = 2*np.pi / (NY * DX)
    dk_z = 2*np.pi / (NZ * DX)
    kx = np.fft.fftfreq(NX) * NX * dk_x  # [0,1,...,N/2-1,-N/2,...,-1]*dk
    ky = np.fft.fftfreq(NY) * NY * dk_y
    kz = np.fft.fftfreq(NZ) * NZ * dk_z
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    KMAG = np.sqrt(KX**2 + KY**2 + KZ**2)

    # kappa = sinc(c0*dt*|k|/2)  [unnormalized]
    kappa = sinc_unnorm(C0 * DT * KMAG / 2.0)

    # FFT of p0
    p0_k = np.fft.fftn(p0)

    # Exact PSTD prediction: p[1]_k = p0_k * cos(c0*dt*|k|)
    p1_exact_k = p0_k * np.cos(C0 * DT * KMAG)
    p1_exact = np.real(np.fft.ifftn(p1_exact_k))

    # Leapfrog prediction (what pykwavers should do):
    #   p[1]_k = p0_k * kappa * (1 - kappa*c0²*dt²*|k|²)
    #          = p0_k * (kappa - kappa²*4*theta²)   [theta = c0*dt*|k|/2]
    theta = C0 * DT * KMAG / 2.0
    p1_leapfrog_k = p0_k * kappa * (1.0 - kappa * 4.0 * theta**2)
    p1_leapfrog = np.real(np.fft.ifftn(p1_leapfrog_k))

    # ── Run pykwavers for 1 step (no PML) ────────────────────────────────────
    grid   = pkw.Grid(NX, NY, NZ, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_initial_pressure(p0.copy())
    sensor_mask = np.ones((NX, NY, NZ), dtype=bool)   # record everywhere
    sensor = pkw.Sensor.from_mask(sensor_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    # No PML for clean physics test
    sim.set_pml_size(0)

    result = sim.run(time_steps=1, dt=DT)
    p1_pkw_flat = np.asarray(result.sensor_data, dtype=np.float64).flatten()
    # sensor_data is ordered by sensor indices (row-major)
    # Reconstruct 3D field from sensor data
    # With all-ones sensor mask, we have NX*NY*NZ sensors
    # The ordering in SensorRecorder matches the input mask iteration order
    # (C-order / row-major for the boolean mask)
    p1_pkw = p1_pkw_flat.reshape((NX, NY, NZ))

    # ── Compare ───────────────────────────────────────────────────────────────
    # At center (source point)
    print(f"\nPressure at center [{cx},{cy},{cz}] after 1 step:")
    print(f"  Exact PSTD:    {p1_exact[cx,cy,cz]:.6e} Pa")
    print(f"  Leapfrog pred: {p1_leapfrog[cx,cy,cz]:.6e} Pa")
    print(f"  pykwavers:     {p1_pkw[cx,cy,cz]:.6e} Pa")

    # At adjacent point
    print(f"\nPressure at [{cx+1},{cy},{cz}] after 1 step:")
    print(f"  Exact PSTD:    {p1_exact[cx+1,cy,cz]:.6e} Pa")
    print(f"  Leapfrog pred: {p1_leapfrog[cx+1,cy,cz]:.6e} Pa")
    print(f"  pykwavers:     {p1_pkw[cx+1,cy,cz]:.6e} Pa")

    # RMS errors
    rms_exact  = np.sqrt(np.mean(p1_exact**2))
    rms_pkw    = np.sqrt(np.mean(p1_pkw**2))
    rms_lf     = np.sqrt(np.mean(p1_leapfrog**2))

    print(f"\nRMS over full 3D grid:")
    print(f"  Exact PSTD:    {rms_exact:.6e} Pa")
    print(f"  Leapfrog pred: {rms_lf:.6e} Pa")
    print(f"  pykwavers:     {rms_pkw:.6e} Pa")

    # Field-level Pearson r
    def rms(a, b):
        a_f = a.flatten(); b_f = b.flatten()
        a_m = a_f - a_f.mean(); b_m = b_f - b_f.mean()
        d = np.sqrt((a_m**2).sum() * (b_m**2).sum())
        return float((a_m * b_m).sum() / d) if d > 1e-30 else 0.0

    r_exact_pkw  = rms(p1_exact, p1_pkw)
    r_exact_lf   = rms(p1_exact, p1_leapfrog)
    r_lf_pkw     = rms(p1_leapfrog, p1_pkw)

    print(f"\nPearson r after 1 step:")
    print(f"  Exact vs pykwavers:    {r_exact_pkw:.6f}")
    print(f"  Exact vs leapfrog:     {r_exact_lf:.6f}")
    print(f"  Leapfrog vs pykwavers: {r_lf_pkw:.6f}")

    # How well does pykwavers match the leapfrog analytical prediction?
    diff_pkw_lf = np.abs(p1_pkw - p1_leapfrog).max()
    diff_pkw_ex = np.abs(p1_pkw - p1_exact).max()
    print(f"\nMax absolute difference:")
    print(f"  |pykwavers - leapfrog| = {diff_pkw_lf:.3e} Pa")
    print(f"  |pykwavers - exact|    = {diff_pkw_ex:.3e} Pa")

    # K-space comparison: where do exact and leapfrog differ most?
    ratio_k = np.abs(p1_leapfrog_k) / (np.abs(p1_exact_k) + 1e-30)
    print(f"\nK-space ratio |leapfrog| / |exact| statistics:")
    print(f"  min: {ratio_k.min():.4f}")
    print(f"  max: {ratio_k.max():.4f}")
    print(f"  mean: {ratio_k.mean():.4f}")

    # Show k-space ratio vs theta at a few key wavenumbers
    print(f"\nK-space ratio at key modes (theta = c0*dt*|k|/2):")
    print(f"  DC (|k|=0):          theta={0:.4f}  ratio={1.0:.4f}  cos(2θ)/leapfrog={1.0:.4f}")
    thetas_test = [0.0, 0.1, 0.2, 0.3, 0.302]  # various theta values
    for th in thetas_test[1:]:
        kap = np.sin(th)/th if th > 1e-10 else 1.0
        lf_val = kap - kap**2 * 4 * th**2
        ex_val = np.cos(2*th)
        print(f"  theta={th:.3f}: kappa={kap:.4f}  leapfrog_p1/p0={lf_val:.4f}  "
              f"exact_p1/p0={ex_val:.4f}  ratio={lf_val/ex_val:.4f}")

    # How many steps does the leapfrog error shift the wavefront?
    # The leapfrog gives p[1] as if the wave were at phase cos(2θ*(1 - delta_n))
    # where delta_n is the effective step count error.
    # cos(2θ*(1-delta_n)) = kappa - kappa²*4θ²
    # For small θ: cos(2θ*(1-delta_n)) ≈ 1 - 2θ²*(1-delta_n)²
    # kappa - kappa²*4θ² ≈ 1 - θ²/6 - 4θ² = 1 - 25θ²/6
    # So: 2*(1-delta_n)² = 25/6 → (1-delta_n)² = 25/12 → 1-delta_n = 1.44
    # → delta_n = -0.44 (not 7!)
    # The 7-step offset must come from SOMEWHERE ELSE.

    print("\nConclusion:")
    print(f"  If pykwavers matches leapfrog: r={r_lf_pkw:.4f} (should be ~1.0)")
    print(f"  If pykwavers matches exact:    r={r_exact_pkw:.4f} (should be ~1.0)")
    print(f"  The discrepancy pykwavers vs exact at step 1 does NOT")
    print(f"  by itself explain the 7-step timing offset in the sensor trace.")

if __name__ == "__main__":
    main()
