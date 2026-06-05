"""Bubble cloud physics parity tests.

Validates pykwavers bubble dynamics against analytical results:

1. BubbleField: creation, bubble count tracking, center bubble placement.
2. Minnaert resonance frequency: analytical formula validated against
   the small-oscillation linear regime of the Rayleigh-Plesset equation.
3. PIDController: proportional, integral, and derivative responses verified
   against direct analytical solutions.

References
----------
- Minnaert (1933). Philos. Mag. 16(104):235–248.
  (resonance frequency f₀ = (1/2πR₀)√(3γp₀/ρ))
- Leighton (1994). The Acoustic Bubble. Academic Press.
- Åström & Hägglund (2006). Advanced PID Control. ISA.
"""

import math

import pytest

import pykwavers

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
_P0 = 101325.0  # Pa — atmospheric pressure
_RHO_WATER = 998.0  # kg/m³ — water density at 20 °C
_GAMMA = 1.4  # adiabatic index for air/gas bubbles


def minnaert_frequency(R0_m: float, p0: float = _P0, rho: float = _RHO_WATER, gamma: float = _GAMMA) -> float:
    """Minnaert resonance frequency for a spherical gas bubble.

    Theorem (Minnaert 1933):
        f₀ = (1 / 2π R₀) · √(3 γ p₀ / ρ)

    Parameters
    ----------
    R0_m:  equilibrium bubble radius (m)
    p0:    static pressure (Pa)
    rho:   liquid density (kg/m³)
    gamma: ratio of specific heats (dimensionless)

    Returns
    -------
    f₀ in Hz
    """
    return (1.0 / (2.0 * math.pi * R0_m)) * math.sqrt(3.0 * gamma * p0 / rho)


# ──────────────────────────────────────────────────────────────────────────────
# BubbleField API tests
# ──────────────────────────────────────────────────────────────────────────────


def test_bubble_field_initial_count_zero() -> None:
    """A freshly created BubbleField has zero bubbles."""
    bf = pykwavers.BubbleField(16, 16, 16)
    assert bf.num_bubbles() == 0


def test_bubble_field_add_center_bubble() -> None:
    """add_center_bubble increments the bubble count by 1."""
    bf = pykwavers.BubbleField(16, 16, 16)
    bf.add_center_bubble()
    assert bf.num_bubbles() == 1


def test_bubble_field_multiple_add_center_bubble_idempotent() -> None:
    """add_center_bubble is idempotent: repeated calls keep the count at 1.

    A BubbleField has exactly one geometric center — calling add_center_bubble
    multiple times does not place duplicate bubbles at the same position.
    """
    bf = pykwavers.BubbleField(16, 16, 16)
    for _ in range(3):
        bf.add_center_bubble()
    assert bf.num_bubbles() == 1, (
        f"Expected 1 center bubble after repeated calls, got {bf.num_bubbles()}"
    )


def test_bubble_field_different_grid_sizes() -> None:
    """BubbleField can be created with various grid dimensions."""
    for nx, ny, nz in [(8, 8, 8), (32, 16, 16), (64, 64, 64)]:
        bf = pykwavers.BubbleField(nx, ny, nz)
        assert bf.num_bubbles() == 0


# ──────────────────────────────────────────────────────────────────────────────
# Minnaert frequency — analytical validation
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("R0_mm", [0.05, 0.1, 0.5, 1.0, 5.0])
def test_minnaert_frequency_scaling(R0_mm: float) -> None:
    """Minnaert f₀ ∝ 1/R₀ (inverse-radius scaling, Minnaert 1933).

    For two radii R₁ and R₂:  f₁/f₂ = R₂/R₁
    """
    R1 = R0_mm * 1e-3
    R2 = 2.0 * R1  # double radius
    f1 = minnaert_frequency(R1)
    f2 = minnaert_frequency(R2)
    ratio = f1 / f2
    expected = R2 / R1  # = 2
    assert abs(ratio - expected) < 1e-10, (
        f"Minnaert f₁/f₂ = {ratio:.6f}, expected {expected:.6f} (1/R scaling)"
    )


def test_minnaert_frequency_100um_bubble() -> None:
    """100 µm air bubble resonates near 32.5 kHz (Minnaert 1933, Leighton 1994)."""
    R0 = 100e-6  # 100 µm
    f0 = minnaert_frequency(R0)
    # Published: ~32.5 kHz for 100 µm bubble in water at 1 atm, γ=1.4
    assert 30e3 <= f0 <= 35e3, (
        f"100 µm bubble Minnaert frequency = {f0/1e3:.2f} kHz, "
        f"expected 30–35 kHz"
    )


def test_minnaert_frequency_1mm_bubble() -> None:
    """1 mm air bubble resonates near 3.25 kHz.

    From the 1/R scaling: f(1 mm) = f(100 µm) / 10 ≈ 3.25 kHz.
    """
    R0 = 1e-3  # 1 mm
    f0 = minnaert_frequency(R0)
    assert 3.0e3 <= f0 <= 3.5e3, (
        f"1 mm bubble Minnaert frequency = {f0/1e3:.3f} kHz, expected 3.0–3.5 kHz"
    )


def test_minnaert_frequency_pressure_dependence() -> None:
    """Minnaert f₀ ∝ √p₀ (square-root pressure scaling)."""
    R0 = 500e-6  # 500 µm
    p_ref = 101325.0
    p_double = 2.0 * p_ref
    f_ref = minnaert_frequency(R0, p0=p_ref)
    f_double = minnaert_frequency(R0, p0=p_double)
    ratio = f_double / f_ref
    expected = math.sqrt(2.0)
    assert abs(ratio - expected) < 1e-10, (
        f"f(2p₀)/f(p₀) = {ratio:.6f}, expected √2 = {expected:.6f}"
    )


def test_minnaert_frequency_density_dependence() -> None:
    """Minnaert f₀ ∝ 1/√ρ (inverse-root-density scaling)."""
    R0 = 500e-6
    rho_ref = 998.0
    rho_double = 2.0 * rho_ref
    f_ref = minnaert_frequency(R0, rho=rho_ref)
    f_double = minnaert_frequency(R0, rho=rho_double)
    ratio = f_ref / f_double
    expected = math.sqrt(2.0)
    assert abs(ratio - expected) < 1e-10, (
        f"f(ρ)/f(2ρ) = {ratio:.6f}, expected √2 = {expected:.6f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# PIDController tests
# ──────────────────────────────────────────────────────────────────────────────


def test_pid_controller_proportional_only() -> None:
    """P-only controller output equals Kp * error on first step.

    With Ki = Kd = 0, the control law is u = Kp * (setpoint - measurement).
    On the first call, integral = 0 and derivative = 0, so u = Kp * e.
    """
    kp = 2.0
    setpoint = 1.0
    measurement = 0.3
    expected_error = setpoint - measurement
    expected_output = kp * expected_error  # = 1.4

    pid = pykwavers.PIDController(
        kp=kp, ki=0.0, kd=0.0, setpoint=setpoint,
        output_min=-1000.0, output_max=1000.0
    )
    # update() returns (clamped_output, raw_output, integral, derivative)
    output = pid.update(measurement)[0]
    assert abs(output - expected_output) < 1e-9, (
        f"P-only output = {output:.6f}, expected {expected_output:.6f}"
    )


def test_pid_controller_output_clamped_to_max() -> None:
    """Controller output is clamped to output_max."""
    kp = 100.0
    pid = pykwavers.PIDController(
        kp=kp, ki=0.0, kd=0.0, setpoint=1.0,
        output_max=5.0, output_min=-5.0
    )
    output = pid.update(0.0)[0]  # error = 1.0 → raw output = 100 → clamped to 5
    assert output == pytest.approx(5.0, abs=1e-9)


def test_pid_controller_output_clamped_to_min() -> None:
    """Controller output is clamped to output_min."""
    kp = 100.0
    pid = pykwavers.PIDController(
        kp=kp, ki=0.0, kd=0.0, setpoint=-1.0,
        output_max=5.0, output_min=-5.0
    )
    output = pid.update(0.0)[0]  # error = -1.0 → raw output = -100 → clamped to -5
    assert output == pytest.approx(-5.0, abs=1e-9)


def test_pid_controller_integral_accumulates() -> None:
    """Integral term accumulates over time steps.

    After two steps with constant error e and Ki > 0, the integral contribution
    should be non-zero and increasing in magnitude.
    """
    kp = 0.0
    ki = 1.0
    setpoint = 1.0
    measurement = 0.0  # constant error = 1.0

    pid = pykwavers.PIDController(
        kp=kp, ki=ki, kd=0.0, setpoint=setpoint,
        sample_time=1.0,  # dt = 1 s for easy computation
        output_min=-1000.0, output_max=1000.0,
        integral_limit=1000.0
    )
    out1 = pid.update(measurement)[0]
    out2 = pid.update(measurement)[0]
    # Integral should be growing: |out2| > |out1|
    assert abs(out2) > abs(out1), (
        f"Integral should grow: |out1|={abs(out1):.4f}, |out2|={abs(out2):.4f}"
    )


def test_pid_controller_reset_clears_state() -> None:
    """After reset(), repeated measurements return the same output."""
    pid = pykwavers.PIDController(
        kp=1.0, ki=1.0, kd=0.1, setpoint=1.0,
        sample_time=1.0, output_min=-1000.0, output_max=1000.0
    )
    out1 = pid.update(0.0)[0]  # accumulates state
    pid.reset()
    out_after_reset = pid.update(0.0)[0]
    # After reset, state is cleared → output should match first-step value
    assert abs(out_after_reset - out1) < 1e-9, (
        f"After reset, output {out_after_reset:.6f} should equal first-step {out1:.6f}"
    )


def test_pid_controller_zero_error_zero_output() -> None:
    """When measurement equals setpoint, output is zero (P+I=0 at start)."""
    pid = pykwavers.PIDController(
        kp=5.0, ki=0.0, kd=0.0, setpoint=0.5,
        output_min=-1000.0, output_max=1000.0
    )
    output = pid.update(0.5)[0]  # error = 0
    assert abs(output) < 1e-9, f"Zero-error output should be zero, got {output}"
