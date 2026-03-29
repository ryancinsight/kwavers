"""Water properties parity: pykwavers vs k-wave-python.

Verifies that pykwavers water property functions reproduce k-wave-python
(Marczak 1997 / IAPWS formulae) to within instrument-level tolerances.

References
----------
- Marczak (1997). J. Acoust. Soc. Am. 102(5):2776–2779. (c(T) polynomial)
- Jones & Harris (1992). J. Chem. Eng. Data 37(4):529–533. (ρ(T))
- Francois & Garrison (1982). J. Acoust. Soc. Am. 72(3):896–907. (α(f,T))
"""

import math

import numpy as np
import pytest

import pykwavers
from kwave.utils.mapgen import (
    water_absorption as kw_water_absorption,
    water_density as kw_water_density,
    water_non_linearity as kw_water_non_linearity,
    water_sound_speed as kw_water_sound_speed,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_LN10 = math.log(10.0)
_CM_PER_M = 100.0


def _kw_absorption_to_np_per_m(kw_dB_per_cm: float) -> float:
    """Convert k-wave absorption (dB/cm at a given freq) → Np/m."""
    # dB/cm → dB/m → Np/m:  α_Np = α_dB * ln(10)/20
    return kw_dB_per_cm * _CM_PER_M * _LN10 / 20.0


# ──────────────────────────────────────────────────────────────────────────────
# Sound speed c(T) — Marczak (1997)
# ──────────────────────────────────────────────────────────────────────────────

# Tolerance from Marczak (1997): ≤ 0.2 m/s over 0–95 °C.
C_ABS_TOL = 0.5  # m/s — slightly generous for floating-point/round-trip

TEMP_POINTS_C = [5.0, 10.0, 20.0, 37.0, 60.0, 80.0]
# k-wave-python water_density is limited to 5–40 °C
TEMP_POINTS_DENSITY_KW = [5.0, 10.0, 20.0, 37.0]


@pytest.mark.parametrize("T", TEMP_POINTS_C)
def test_water_sound_speed_matches_kwave(T: float) -> None:
    """pykwavers.water_sound_speed(T) matches k-wave-python to ≤ 0.5 m/s."""
    pyk = pykwavers.water_sound_speed(T)
    kw = kw_water_sound_speed(T)
    assert abs(pyk - kw) <= C_ABS_TOL, (
        f"T={T} °C: pykwavers={pyk:.4f} m/s, kwave={kw:.4f} m/s, "
        f"diff={abs(pyk-kw):.4f} m/s > {C_ABS_TOL} m/s"
    )


def test_water_sound_speed_vector() -> None:
    """Vectorised pykwavers call produces same results as scalar loop."""
    temps = np.array(TEMP_POINTS_C)
    pyk_vec = np.array([pykwavers.water_sound_speed(t) for t in temps])
    kw_vec = kw_water_sound_speed(temps)
    np.testing.assert_allclose(pyk_vec, kw_vec, atol=C_ABS_TOL,
        err_msg="Vectorised sound speed mismatch vs k-wave-python")


def test_water_sound_speed_physical_range() -> None:
    """Sound speed in water lies in the physical range [1400, 1600] m/s for 0–95 °C."""
    for T in TEMP_POINTS_C:
        c = pykwavers.water_sound_speed(T)
        assert 1400.0 <= c <= 1600.0, f"c({T} °C) = {c:.1f} m/s outside physical range"


def test_water_sound_speed_maximum_near_74c() -> None:
    """Water sound speed peaks near 74 °C (Marczak 1997)."""
    c60 = pykwavers.water_sound_speed(60.0)
    c74 = pykwavers.water_sound_speed(74.0)
    c80 = pykwavers.water_sound_speed(80.0)
    assert c74 >= c60, "c(74) should be ≥ c(60)"
    assert c74 >= c80, "c(74) should be ≥ c(80) — near maximum"


# ──────────────────────────────────────────────────────────────────────────────
# Density ρ(T) — Jones & Harris (1992)
# ──────────────────────────────────────────────────────────────────────────────

RHO_ABS_TOL = 0.5  # kg/m³


@pytest.mark.parametrize("T", TEMP_POINTS_DENSITY_KW)
def test_water_density_matches_kwave(T: float) -> None:
    """pykwavers.water_density(T) matches k-wave-python to ≤ 0.5 kg/m³.

    k-wave-python water_density is valid for T ∈ [5, 40] °C only.
    """
    pyk = pykwavers.water_density(T)
    kw = kw_water_density(T)
    assert abs(pyk - kw) <= RHO_ABS_TOL, (
        f"T={T} °C: pykwavers={pyk:.4f} kg/m³, kwave={kw:.4f} kg/m³, "
        f"diff={abs(pyk-kw):.4f} > {RHO_ABS_TOL}"
    )


def test_water_density_decreases_with_temperature() -> None:
    """Water density decreases monotonically above 4 °C."""
    temps = [5.0, 10.0, 20.0, 37.0, 60.0, 80.0]
    densities = [pykwavers.water_density(T) for T in temps]
    for i in range(len(densities) - 1):
        assert densities[i] > densities[i + 1], (
            f"ρ({temps[i]}) = {densities[i]:.4f} should be > "
            f"ρ({temps[i+1]}) = {densities[i+1]:.4f}"
        )


def test_water_density_near_1000_at_20c() -> None:
    """Water density near 20 °C is ~998 kg/m³ (well-known physical value)."""
    rho = pykwavers.water_density(20.0)
    assert 996.0 <= rho <= 1000.0, f"ρ(20 °C) = {rho:.2f} kg/m³ outside expected range"


# ──────────────────────────────────────────────────────────────────────────────
# Absorption α(f, T) — Francois & Garrison (1982)
# ──────────────────────────────────────────────────────────────────────────────

# pykwavers returns Np/m at the given frequency.
# k-wave-python returns dB/cm at the given frequency (f in MHz).
# Conversion: Np/m = dB/cm * ln(10)/20 * 100
ALPHA_REL_TOL = 0.01  # 1 % relative tolerance

FREQ_POINTS_HZ = [0.5e6, 1.0e6, 2.0e6, 5.0e6]
TEMP_POINTS_ALPHA = [10.0, 20.0, 37.0]


@pytest.mark.parametrize("f_hz", FREQ_POINTS_HZ)
@pytest.mark.parametrize("T", TEMP_POINTS_ALPHA)
def test_water_absorption_matches_kwave(f_hz: float, T: float) -> None:
    """pykwavers.water_absorption(f_Hz, T) matches kwave (unit-converted) to ≤ 1%."""
    pyk = pykwavers.water_absorption(f_hz, T)
    kw_dB_cm = kw_water_absorption(f_hz / 1e6, T)
    kw_np_m = _kw_absorption_to_np_per_m(kw_dB_cm)

    rel_err = abs(pyk - kw_np_m) / max(kw_np_m, 1e-30)
    assert rel_err <= ALPHA_REL_TOL, (
        f"f={f_hz/1e6:.1f} MHz, T={T} °C: "
        f"pykwavers={pyk:.4e} Np/m, kwave_converted={kw_np_m:.4e} Np/m, "
        f"relative error={rel_err:.4%} > {ALPHA_REL_TOL:.0%}"
    )


def test_water_absorption_increases_with_frequency() -> None:
    """Absorption increases with frequency (power-law, y≈2 for water)."""
    T = 20.0
    alphas = [pykwavers.water_absorption(f, T) for f in FREQ_POINTS_HZ]
    for i in range(len(alphas) - 1):
        assert alphas[i] < alphas[i + 1], (
            f"Absorption should increase with frequency: "
            f"α({FREQ_POINTS_HZ[i]/1e6:.1f} MHz) = {alphas[i]:.3e} ≮ "
            f"α({FREQ_POINTS_HZ[i+1]/1e6:.1f} MHz) = {alphas[i+1]:.3e}"
        )


def test_water_absorption_power_law_exponent_near_2() -> None:
    """Water absorption follows α ~ f^y with y ≈ 2 (Francois & Garrison 1982)."""
    T = 20.0
    f1, f2 = 1e6, 2e6
    a1 = pykwavers.water_absorption(f1, T)
    a2 = pykwavers.water_absorption(f2, T)
    y_measured = math.log(a2 / a1) / math.log(f2 / f1)
    assert 1.8 <= y_measured <= 2.2, (
        f"Water absorption exponent y = {y_measured:.3f}, expected ≈ 2.0"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Nonlinearity B/A — Beyer (1960), Hamilton & Blackstock (1998)
# ──────────────────────────────────────────────────────────────────────────────

BA_ABS_TOL = 0.01  # B/A is dimensionless; ±0.01 is sub-percent


@pytest.mark.parametrize("T", TEMP_POINTS_C)
def test_water_nonlinearity_matches_kwave(T: float) -> None:
    """pykwavers.water_nonlinearity(T) matches k-wave-python to ≤ 0.01."""
    pyk = pykwavers.water_nonlinearity(T)
    kw = kw_water_non_linearity(T)
    assert abs(pyk - kw) <= BA_ABS_TOL, (
        f"T={T} °C: pykwavers={pyk:.5f}, kwave={kw:.5f}, "
        f"diff={abs(pyk-kw):.5f} > {BA_ABS_TOL}"
    )


def test_water_nonlinearity_increases_with_temperature() -> None:
    """B/A increases monotonically with temperature for water."""
    ba_20 = pykwavers.water_nonlinearity(20.0)
    ba_60 = pykwavers.water_nonlinearity(60.0)
    assert ba_60 > ba_20, f"B/A({60} °C)={ba_60:.4f} should be > B/A(20 °C)={ba_20:.4f}"


def test_water_nonlinearity_physical_range() -> None:
    """B/A for water lies in [3.5, 7.0] over 0–95 °C (Hamilton & Blackstock 1998)."""
    for T in TEMP_POINTS_C:
        ba = pykwavers.water_nonlinearity(T)
        assert 3.5 <= ba <= 7.0, f"B/A({T} °C) = {ba:.4f} outside [3.5, 7.0]"
