"""
Westervelt nonlinear acoustic propagation parity test.

Validates that kwavers' Westervelt nonlinear source term produces harmonic
amplitudes consistent with the Fubini solution (exact analytical result for
lossless, 1-D plane-wave propagation before shock formation).

Algorithm
---------
Fubini (1935) showed that for a sinusoidal plane wave of amplitude p₀,
frequency f₀, and nonlinearity coefficient β = 1 + B/(2A) propagating in a
lossless medium with density ρ₀ and sound speed c₀, the pressure at distance
x before the shock distance x_s = ρ₀c₀³/(β p₀ ω₀) is:

    p(x, t) = 2p₀ Σ_{n=1}^{∞}  Jₙ(nΓ) / (nΓ)  sin(n(ω₀t − k₀x))

where Γ = β p₀ k₀ x / (ρ₀ c₀²) is the Gol'dberg number and Jₙ is the
Bessel function of the first kind.

At x = x_s/2 (half shock distance, Γ = 1/2), the second harmonic amplitude
relative to the fundamental is:

    |p₂| / |p₁| = J₂(1) / J₁(1)  ≈ 0.5769

Test protocol
-------------
1. Run a 1-D (effectively 3-D with Ny=Nz=1) FDTD kwavers simulation
   with the Westervelt nonlinear solver enabled.
2. Record pressure at x = x_s/2 after steady-state is reached.
3. Extract fundamental and second-harmonic amplitudes via FFT.
4. Compare the ratio p₂/p₁ to the Fubini prediction.

Thresholds
----------
- |p₂/p₁ − Fubini| / Fubini < 0.20  (20% relative tolerance; broader
  than production targets to accommodate discrete-grid dispersion and
  PML-induced amplitude drift in 3-D)

References
----------
Fubini, E. (1935). Anomalie nella propagazione di onde acustiche di grande
ampiezza. Alta Frequenza, 4, 530–581.

Hamilton, M. F., & Blackstock, D. T. (Eds.). (1998). Nonlinear Acoustics.
Academic Press. Chapter 4 (Lossless Plane Waves).

Westervelt, P. J. (1963). Parametric Acoustic Array. Journal of the
Acoustical Society of America, 35(4), 535–537.
"""
import numpy as np
import pytest
from scipy.special import jv as bessel_j

try:
    import pykwavers as kw
    PYKWAVERS_AVAILABLE = True
except ImportError:
    PYKWAVERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PYKWAVERS_AVAILABLE,
    reason="pykwavers required for Westervelt parity test"
)

# ── Physical constants ──────────────────────────────────────────────────────

C0 = 1500.0        # m/s  — sound speed
RHO0 = 1000.0      # kg/m³ — density
B_OVER_A = 5.0     # B/A for water at 20°C (Beyer 1960)
BETA = 1.0 + B_OVER_A / 2.0   # nonlinearity coefficient β = 1 + B/(2A) = 3.5

F0 = 0.5e6         # Hz — fundamental frequency
OMEGA0 = 2.0 * np.pi * F0
K0 = OMEGA0 / C0   # wave number [1/m]

P0_AMPLITUDE = 1e4  # Pa  — source amplitude (weak nonlinearity regime)

# Shock distance x_s = ρ₀ c₀³ / (β p₀ ω₀)
X_SHOCK = RHO0 * C0 ** 3 / (BETA * P0_AMPLITUDE * OMEGA0)

# Test point at half shock distance
X_TEST = 0.5 * X_SHOCK  # [m]  — ≈ 16 mm for typical water/HIFU parameters

# Gol'dberg number Γ at x = x_s/2
GAMMA_TEST = 0.5   # = β p₀ k₀ x / (ρ₀ c₀²) = x/x_s × 1 = 0.5

# Fubini harmonic ratios at Γ = 0.5:
# p_n = 2 p₀ Jₙ(nΓ) / (nΓ)
FUBINI_P1_NORM = 2.0 * bessel_j(1, 1.0 * GAMMA_TEST) / (1.0 * GAMMA_TEST)
FUBINI_P2_NORM = 2.0 * bessel_j(2, 2.0 * GAMMA_TEST) / (2.0 * GAMMA_TEST)
FUBINI_RATIO = FUBINI_P2_NORM / FUBINI_P1_NORM   # ≈ 0.577


# ── Simulation helpers ──────────────────────────────────────────────────────

def run_westervelt_simulation():
    """
    Run a kwavers 1-D Westervelt simulation and return the sensor time series
    at x = X_TEST.

    Returns
    -------
    t : ndarray  — time array [s]
    p : ndarray  — pressure time series at sensor [Pa]
    """
    # Grid: 1-D propagation along x; Ny=Nz=1 (quasi-1D)
    dx = X_SHOCK / 100.0          # ~10 cells before half-shock distance
    Nx = max(int(X_TEST / dx) + 20, 64)  # enough cells to include X_TEST + PML
    Ny = Nz = 1

    # Time stepping: CFL ≤ 0.3 for stability
    dt = 0.3 * dx / C0
    # Run for ~10 acoustic periods to reach steady state
    Nt = int(10.0 / (F0 * dt))

    grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)

    # Homogeneous medium with nonlinear parameter (B/A = 5.0)
    medium = kw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        nonlinearity=B_OVER_A,
    )

    # Source: sinusoidal plane wave at x=0
    t_arr = np.arange(Nt) * dt
    source_signal = P0_AMPLITUDE * np.sin(OMEGA0 * t_arr)
    src_mask = np.zeros((Nx, Ny, Nz))
    src_mask[0, 0, 0] = 1.0

    source = kw.Source.from_mask(
        src_mask.astype(np.float64), source_signal, F0, mode="additive"
    )

    # Sensor at x = X_TEST
    sen_ix = int(X_TEST / dx)
    sen_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sen_mask[min(sen_ix, Nx - 1), 0, 0] = True
    sensor = kw.Sensor.from_mask(sen_mask)

    # Run FDTD with nonlinear Westervelt solver
    sim = kw.Simulation(
        grid, medium, source, sensor,
        solver=kw.SolverType.FDTD,
    )
    sim.set_nonlinear(True)   # enable Westervelt source term
    result = sim.run(time_steps=Nt, dt=dt)

    p = np.array(result.sensor_data).flatten()
    return t_arr, p


def extract_harmonic_amplitudes(t, p, f0, n_harmonics=3, n_periods_discard=5):
    """
    Extract amplitudes of harmonics 1..n_harmonics from time series p(t).

    Uses a rectangular window over the last (Nt - n_periods_discard × T₀) samples
    to discard transient startup.

    Returns amplitudes in Pa.
    """
    T0 = 1.0 / f0
    dt = t[1] - t[0]
    n_discard = int(n_periods_discard * T0 / dt)
    p_ss = p[n_discard:]   # steady-state portion

    N = len(p_ss)
    P = np.fft.rfft(p_ss) * 2.0 / N  # one-sided, peak amplitude
    freqs = np.fft.rfftfreq(N, d=dt)

    amplitudes = []
    for n in range(1, n_harmonics + 1):
        f_target = n * f0
        idx = np.argmin(np.abs(freqs - f_target))
        amplitudes.append(np.abs(P[idx]))

    return amplitudes


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_westervelt_second_harmonic_fubini():
    """
    Westervelt second-harmonic amplitude relative to fundamental must match
    the Fubini solution within 20% at x = x_s/2.

    The Fubini solution at Γ=0.5:
        p₂/p₁ = J₂(1.0) / J₁(0.5) × [J₁(0.5)/J₁(0.5)]⁻¹
              = J₂(2×0.5)/(2×0.5) / [J₁(1×0.5)/(1×0.5)]
              ≈ 0.577
    """
    t, p = run_westervelt_simulation()
    amps = extract_harmonic_amplitudes(t, p, F0)
    p1, p2 = amps[0], amps[1]

    # Avoid divide-by-zero if simulation produces all zeros
    assert p1 > 1.0, f"Fundamental amplitude too small: {p1:.3e} Pa (source P0={P0_AMPLITUDE:.3e})"

    ratio = p2 / p1
    rel_err = abs(ratio - FUBINI_RATIO) / FUBINI_RATIO

    assert rel_err < 0.20, (
        f"Second-harmonic ratio {ratio:.4f} deviates {rel_err*100:.1f}% "
        f"from Fubini prediction {FUBINI_RATIO:.4f} at x=x_s/2 (Γ=0.5)"
    )


@pytest.mark.slow
def test_westervelt_nonlinear_enabled_creates_harmonics():
    """
    Sanity check: Westervelt solver must produce visible second harmonic (p₂/p₁ > 0.01).
    A linear solver would give p₂ ≈ 0.
    """
    t, p = run_westervelt_simulation()
    amps = extract_harmonic_amplitudes(t, p, F0)
    p1, p2 = amps[0], amps[1]

    assert p1 > 0.0, "Fundamental must be non-zero"
    ratio = p2 / (p1 + 1e-30)
    assert ratio > 0.01, (
        f"Second harmonic ratio {ratio:.4f} is too small — "
        "nonlinear Westervelt term may not be active"
    )
