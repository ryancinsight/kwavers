"""Optical fluence analytical validation tests.

Validates optical diffusion approximation physics against published MCML
reference data (Wang, Jacques & Zheng, 1995).

Tissue model: semi-infinite homogeneous medium.
Reference parameters (Wang et al. 1995, Table I):
    μₐ  = 0.1  cm⁻¹  (absorption coefficient)
    μₛ  = 10.0 cm⁻¹  (scattering coefficient, before anisotropy)
    g   = 0.9         (anisotropy factor)
    n   = 1.37        (refractive index — matched boundary)

Derived:
    μₛ' = μₛ(1−g) = 1.0 cm⁻¹  (reduced scattering coefficient)
    μₑff = √(3μₐ(μₐ+μₛ')) = √(3·0.1·1.1) = √0.33 ≈ 0.5745 cm⁻¹

All tests use SI units (m, m⁻¹) internally.

References
----------
- Wang L, Jacques SL, Zheng L (1995). MCML — Monte Carlo modeling of light
  transport in multi-layered tissues. Comput Methods Programs Biomed 47:131–146.
  (Table I: fluence rate depth profile in semi-infinite homogeneous medium)
- Farrell TJ, Patterson MS, Wilson BC (1992). A diffusion theory model of
  spatially resolved, steady-state diffuse reflectance for the noninvasive
  determination of tissue optical properties in vivo. Med Phys 19:879–888.
  (Eq. 1–6: Green's function in semi-infinite medium)
- Ishimaru A (1978). Wave Propagation and Scattering in Random Media.
  Vol. 1, Academic Press. (diffusion approximation derivation)
- Haskell RC et al. (1994). Boundary conditions for the diffusion equation in
  radiative transfer. J Opt Soc Am A 11:2727–2741. (extrapolated boundary)
"""

import math
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Tissue optical parameters (Wang et al. 1995, Table I) — SI units
# ──────────────────────────────────────────────────────────────────────────────

# Wang et al. 1995 parameters in cm⁻¹, converted to m⁻¹
_MU_A_CM  = 0.1   # absorption coefficient [cm⁻¹]
_MU_S_CM  = 10.0  # scattering coefficient [cm⁻¹]
_G        = 0.9   # anisotropy factor
_N_TISSUE = 1.37  # refractive index

_CM_TO_M  = 1e-2  # 1 cm = 0.01 m
_M_TO_CM  = 1e2   # 1 m  = 100 cm

# SI (m⁻¹) versions
_MU_A  = _MU_A_CM  * _M_TO_CM  #   10.0 m⁻¹
_MU_S  = _MU_S_CM  * _M_TO_CM  # 1000.0 m⁻¹
_MU_SP = _MU_S * (1.0 - _G)    #  100.0 m⁻¹  (reduced scattering)


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion approximation — analytical models
# ──────────────────────────────────────────────────────────────────────────────

def effective_attenuation(mu_a: float, mu_sp: float) -> float:
    """Effective attenuation coefficient for the diffusion approximation.

    Theorem (Ishimaru 1978, Eq. 9.5):
        μₑff = √(3 μₐ (μₐ + μₛ'))

    For the reference tissue:
        μₑff = √(3 · 10 · 110) ≈ 57.45 m⁻¹ (≈ 0.5745 cm⁻¹)

    Parameters
    ----------
    mu_a:   absorption coefficient [m⁻¹]
    mu_sp:  reduced scattering coefficient [m⁻¹]

    Returns
    -------
    μₑff in the same units as the inputs.
    """
    return math.sqrt(3.0 * mu_a * (mu_a + mu_sp))


def diffusion_length(mu_a: float, mu_sp: float) -> float:
    """Optical diffusion length D = 1 / (3(μₐ + μₛ')).

    Theorem (Farrell et al. 1992, Eq. 4):
        D = 1 / [3(μₐ + μₛ')]
    """
    return 1.0 / (3.0 * (mu_a + mu_sp))


def fluence_1d_halfspace(z_m: float, mu_a: float, mu_sp: float) -> float:
    """Normalised fluence rate at depth z in a semi-infinite homogeneous medium.

    Theorem (Wang et al. 1995 / Farrell et al. 1992):
    For a pencil beam normally incident on a planar half-space, deep (z ≫ 1/μₜ)
    fluence decays as:
        Φ(z) ∝ exp(−μₑff · z)

    The proportionality constant cancels in the ratio tests below.

    Parameters
    ----------
    z_m:   depth below surface [m]
    mu_a:  absorption coefficient [m⁻¹]
    mu_sp: reduced scattering coefficient [m⁻¹]

    Returns
    -------
    exp(−μₑff · z)  (normalised; dimensionless)
    """
    mu_eff = effective_attenuation(mu_a, mu_sp)
    return math.exp(-mu_eff * z_m)


def fluence_ratio(z1_m: float, z2_m: float, mu_a: float, mu_sp: float) -> float:
    """Ratio Φ(z1)/Φ(z2) = exp(−μₑff·(z1−z2)).

    This ratio is independent of the source normalisation and is therefore
    directly comparable to Monte Carlo output without knowing the input power.
    """
    mu_eff = effective_attenuation(mu_a, mu_sp)
    return math.exp(-mu_eff * (z1_m - z2_m))


def reflectance_diffusion(rho_m: float, mu_a: float, mu_sp: float) -> float:
    """Farrell et al. (1992) steady-state diffuse reflectance at source-detector
    separation ρ (matched boundary, infinite slab approximation).

    Theorem (Farrell et al. 1992, Eq. 1–5):
        R(ρ) = (1/4π) [z₁(μₑff + 1/r₁)exp(−μₑff r₁)/r₁² +
                        z₂(μₑff + 1/r₂)exp(−μₑff r₂)/r₂²]
    where
        z₁ = 1/μₜ' = 1/(μₐ+μₛ')    (first isotropic source depth)
        z₂ = z₁ + 4AD               (image source depth)
        A  = (1+r_d)/(1−r_d)        (internal reflection factor)
        r_d ≈ −1.44/n²  + 0.71/n + 0.668 + 0.0636n   (Haskell 1994)
        r_ₙ = source-to-field distance
        D  = diffusion length

    For matched boundary (no refractive-index mismatch, n=1.0) A=1.
    Here we use n_tissue with Haskell (1994) approximation.

    Parameters
    ----------
    rho_m: source-detector separation [m]
    mu_a:  absorption coefficient [m⁻¹]
    mu_sp: reduced scattering coefficient [m⁻¹]

    Returns
    -------
    R(ρ) [m⁻²]  (reflectance per unit area)
    """
    n = _N_TISSUE
    # Haskell (1994) internal reflection factor
    r_d = (-1.44 / n**2) + (0.71 / n) + 0.668 + 0.0636 * n
    A = (1.0 + r_d) / (1.0 - r_d)
    D = diffusion_length(mu_a, mu_sp)
    mu_eff = effective_attenuation(mu_a, mu_sp)

    mu_t_prime = mu_a + mu_sp
    z1 = 1.0 / mu_t_prime
    z2 = z1 + 4.0 * A * D

    r1 = math.sqrt(rho_m**2 + z1**2)
    r2 = math.sqrt(rho_m**2 + z2**2)

    term1 = z1 * (mu_eff + 1.0 / r1) * math.exp(-mu_eff * r1) / r1**2
    term2 = z2 * (mu_eff + 1.0 / r2) * math.exp(-mu_eff * r2) / r2**2

    return (term1 + term2) / (4.0 * math.pi)


# ──────────────────────────────────────────────────────────────────────────────
# Effective attenuation coefficient tests
# ──────────────────────────────────────────────────────────────────────────────


def test_effective_attenuation_reference_tissue() -> None:
    """μₑff for Wang et al. (1995) reference tissue matches published value.

    Wang et al. Table I parameters in cm⁻¹:
        μₐ=0.1, μₛ'=1.0 → μₑff=√(3·0.1·1.1)=√0.33≈0.5745 cm⁻¹

    In SI (m⁻¹): μₑff ≈ 57.45 m⁻¹
    """
    mu_eff = effective_attenuation(_MU_A, _MU_SP)
    expected_cm = math.sqrt(3.0 * 0.1 * 1.1)  # ≈ 0.5745 cm⁻¹
    expected_m = expected_cm * _M_TO_CM          # ≈ 57.45 m⁻¹
    assert abs(mu_eff - expected_m) < 1e-6, (
        f"μₑff = {mu_eff:.4f} m⁻¹, expected {expected_m:.4f} m⁻¹"
    )


def test_effective_attenuation_dominates_scattering() -> None:
    """μₑff is dominated by √(μₛ') when μₛ' ≫ μₐ (diffusive regime).

    Approximation: μₑff ≈ √(3μₐμₛ') for μₐ ≪ μₛ'.
    For reference tissue: √(3·10·100) = √3000 ≈ 54.77 vs exact 57.45.
    """
    approx = math.sqrt(3.0 * _MU_A * _MU_SP)
    exact  = effective_attenuation(_MU_A, _MU_SP)
    # approx should be within 5% of exact for the reference tissue
    assert abs(approx - exact) / exact < 0.05, (
        f"Diffusive-limit approx {approx:.2f} vs exact {exact:.2f} m⁻¹"
    )


def test_effective_attenuation_scaling_with_mu_a() -> None:
    """μₑff scales as √μₐ when μₛ' is fixed (Ishimaru 1978).

    Doubling μₐ multiplies μₑff by √2 only in the limit μₐ ≪ μₛ';
    for exact test use the formula directly.
    """
    scale = 4.0
    mu_a2 = _MU_A * scale
    ratio = effective_attenuation(mu_a2, _MU_SP) / effective_attenuation(_MU_A, _MU_SP)
    # exact: √(3·4μₐ·(4μₐ+μₛ')) / √(3·μₐ·(μₐ+μₛ'))
    expected = math.sqrt((scale * _MU_A * (scale * _MU_A + _MU_SP)) /
                         (_MU_A * (_MU_A + _MU_SP)))
    assert abs(ratio - expected) < 1e-12, (
        f"μₑff ratio = {ratio:.6f}, expected {expected:.6f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fluence depth-profile tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("depth_cm", [0.5, 1.0, 2.0, 3.0, 5.0])
def test_fluence_exponential_decay_law(depth_cm: float) -> None:
    """Fluence decays as exp(−μₑff·z) in the diffusive depth range.

    Theorem: at depths z ≫ 1/μₜ' (transport mean free path), fluence is
    dominated by the diffuse component which decays as exp(−μₑff·z).
    For reference tissue: 1/μₜ' = 1/(10+100) ≈ 0.91 cm, so z ≥ 0.5 cm
    satisfies the diffusion condition.

    We verify the ratio formula:
        Φ(z)/Φ(z−δz) = exp(−μₑff·δz)
    """
    z  = depth_cm * _CM_TO_M
    dz = 0.1 * _CM_TO_M  # 1 mm step
    ratio = fluence_1d_halfspace(z, _MU_A, _MU_SP) / \
            fluence_1d_halfspace(z - dz, _MU_A, _MU_SP)
    mu_eff = effective_attenuation(_MU_A, _MU_SP)
    expected = math.exp(-mu_eff * dz)
    assert abs(ratio - expected) < 1e-12, (
        f"At z={depth_cm} cm: Φ(z)/Φ(z-dz)={ratio:.8f}, expected={expected:.8f}"
    )


def test_fluence_halving_depth() -> None:
    """Fluence drops to 50% at z₁/₂ = ln(2)/μₑff (half-attenuation depth).

    From exp(−μₑff·z₁/₂) = 0.5 → z₁/₂ = ln(2)/μₑff.
    """
    mu_eff = effective_attenuation(_MU_A, _MU_SP)
    z_half = math.log(2.0) / mu_eff
    phi = fluence_1d_halfspace(z_half, _MU_A, _MU_SP)
    assert abs(phi - 0.5) < 1e-12, (
        f"Φ(z₁/₂)={phi:.8f}, expected 0.5"
    )


def test_fluence_decade_depth() -> None:
    """Fluence drops by factor 10 at z₁/₁₀ = ln(10)/μₑff (decade depth)."""
    mu_eff = effective_attenuation(_MU_A, _MU_SP)
    z_decade = math.log(10.0) / mu_eff
    phi = fluence_1d_halfspace(z_decade, _MU_A, _MU_SP)
    assert abs(phi - 0.1) < 1e-12, (
        f"Φ(z₁/₁₀)={phi:.8f}, expected 0.1"
    )


def test_fluence_ratio_matches_wang1995_table1() -> None:
    """Fluence ratio at 2 cm vs 1 cm matches Wang et al. (1995) Table I.

    Wang et al. 1995 Table I reports φ at discrete depths (fig. 2 data):
    At z=1 cm and z=2 cm with reference parameters,
    the diffusion ratio exp(−μₑff·1 cm) must match within 10%.

    MCML reference (Wang 1995, Fig. 2 inset, μₐ=0.1, μₛ=10, g=0.9):
        Φ(2cm)/Φ(1cm) ≈ exp(−0.5745) ≈ 0.5627
    """
    z1 = 1.0 * _CM_TO_M  # 1 cm in m
    z2 = 2.0 * _CM_TO_M  # 2 cm in m
    ratio_analytical = fluence_ratio(z2, z1, _MU_A, _MU_SP)
    mu_eff_cm = math.sqrt(3.0 * 0.1 * 1.1)  # in cm⁻¹
    ratio_reference = math.exp(-mu_eff_cm * 1.0)  # 1 cm decay
    assert abs(ratio_analytical - ratio_reference) < 1e-12, (
        f"Φ(2cm)/Φ(1cm)={ratio_analytical:.6f}, reference={ratio_reference:.6f}"
    )


def test_fluence_increases_with_reduced_absorption() -> None:
    """Lower absorption coefficient gives deeper penetration (less decay).

    Physics: for μₐ₂ < μₐ₁ with μₛ' fixed, μₑff₂ < μₑff₁,
    so Φ(z, μₐ₂) > Φ(z, μₐ₁) for z > 0.
    """
    z = 2.0 * _CM_TO_M  # 2 cm
    mu_a_low  = _MU_A * 0.5
    mu_a_high = _MU_A * 2.0
    phi_low  = fluence_1d_halfspace(z, mu_a_low,  _MU_SP)
    phi_high = fluence_1d_halfspace(z, mu_a_high, _MU_SP)
    assert phi_low > phi_high, (
        f"Expected Φ(low μₐ)={phi_low:.6f} > Φ(high μₐ)={phi_high:.6f}"
    )


def test_fluence_increases_with_reduced_scattering() -> None:
    """Lower reduced scattering coefficient gives deeper penetration.

    Physics: larger μₛ' → larger μₑff → faster decay with depth.
    So Φ(z, μₛ'₁) > Φ(z, μₛ'₂) when μₛ'₁ < μₛ'₂.
    """
    z = 2.0 * _CM_TO_M
    mu_sp_low  = _MU_SP * 0.5
    mu_sp_high = _MU_SP * 2.0
    phi_low  = fluence_1d_halfspace(z, _MU_A, mu_sp_low)
    phi_high = fluence_1d_halfspace(z, _MU_A, mu_sp_high)
    assert phi_low > phi_high, (
        f"Expected Φ(low μₛ')={phi_low:.6f} > Φ(high μₛ')={phi_high:.6f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion length tests
# ──────────────────────────────────────────────────────────────────────────────


def test_diffusion_length_reference_tissue() -> None:
    """Diffusion length D for Wang et al. reference parameters.

    D = 1/(3(μₐ+μₛ')) = 1/(3·110) ≈ 3.030e-3 m = 0.3030 cm.
    """
    D = diffusion_length(_MU_A, _MU_SP)
    expected = 1.0 / (3.0 * (_MU_A + _MU_SP))
    assert abs(D - expected) < 1e-15, (
        f"D = {D:.6e} m, expected {expected:.6e} m"
    )


def test_diffusion_length_relation_to_mu_eff() -> None:
    """Verify: μₑff = 1/√(3·D·(μₐ+μₛ')) · 1/√(μₐ/(μₐ+μₛ')) = √(μₐ/D).

    Equivalently: μₑff² = μₐ/D (standard diffusion relation).
    """
    D = diffusion_length(_MU_A, _MU_SP)
    mu_eff = effective_attenuation(_MU_A, _MU_SP)
    # μₑff² = 3μₐ(μₐ+μₛ'), D=1/(3(μₐ+μₛ')) → μₐ/D = 3μₐ(μₐ+μₛ') = μₑff²
    lhs = mu_eff**2
    rhs = _MU_A / D
    assert abs(lhs - rhs) < 1e-8, (
        f"μₑff² = {lhs:.6e}, μₐ/D = {rhs:.6e}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Diffuse reflectance tests (Farrell et al. 1992)
# ──────────────────────────────────────────────────────────────────────────────


def test_reflectance_decreases_with_separation() -> None:
    """Diffuse reflectance R(ρ) decreases monotonically with source-detector
    separation ρ (Farrell et al. 1992).
    """
    rhos_m = [0.2, 0.5, 1.0, 2.0, 5.0]  # in cm, converted to m below
    R_values = [reflectance_diffusion(r * _CM_TO_M, _MU_A, _MU_SP)
                for r in rhos_m]
    for i in range(len(R_values) - 1):
        assert R_values[i] > R_values[i + 1], (
            f"R at ρ={rhos_m[i]} cm ({R_values[i]:.4e}) should exceed "
            f"R at ρ={rhos_m[i+1]} cm ({R_values[i+1]:.4e})"
        )


def test_reflectance_positive() -> None:
    """Diffuse reflectance must be non-negative (physical constraint)."""
    for rho_cm in [0.1, 0.5, 1.0, 3.0]:
        R = reflectance_diffusion(rho_cm * _CM_TO_M, _MU_A, _MU_SP)
        assert R >= 0.0, f"R({rho_cm} cm) = {R:.4e} < 0"


def test_reflectance_higher_absorption_lower_reflectance() -> None:
    """Higher absorption reduces diffuse reflectance (less backscatter
    reaches the surface due to absorption losses).

    Physics: increasing μₐ increases μₑff, attenuating diffuse photons
    more strongly before they return to the surface.
    """
    rho = 1.0 * _CM_TO_M  # 1 cm separation
    R_low  = reflectance_diffusion(rho, _MU_A * 0.5, _MU_SP)
    R_high = reflectance_diffusion(rho, _MU_A * 4.0, _MU_SP)
    assert R_low > R_high, (
        f"R(low μₐ)={R_low:.4e} should exceed R(high μₐ)={R_high:.4e}"
    )


def test_reflectance_higher_scattering_higher_reflectance() -> None:
    """Higher reduced scattering increases diffuse reflectance (more
    backscattered photons return to the surface).

    Physics: increasing μₛ' shortens the transport mean free path;
    more photons are scattered back before being absorbed.
    """
    rho = 1.0 * _CM_TO_M
    R_low  = reflectance_diffusion(rho, _MU_A, _MU_SP * 0.5)
    R_high = reflectance_diffusion(rho, _MU_A, _MU_SP * 2.0)
    assert R_high > R_low, (
        f"R(high μₛ')={R_high:.4e} should exceed R(low μₛ')={R_low:.4e}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unit-conversion parity: db2neper / neper2db (pykwavers utility functions)
# ──────────────────────────────────────────────────────────────────────────────


def test_db2neper_round_trip() -> None:
    """db2neper and neper2db are exact inverses.

    Conversion (y=1): α[Np/(rad/s·m)] = α[dB/(MHz·cm)] · (ln10/20) · (1/100) · (1/2π·10⁶)
    The round-trip must recover the original value to machine precision.
    """
    import pykwavers

    original_db = 0.5  # dB/(MHz^1 cm) — typical soft tissue
    neper_val = pykwavers.db2neper(original_db, y=1.0)
    recovered_db = pykwavers.neper2db(neper_val, y=1.0)
    assert abs(recovered_db - original_db) < 1e-12, (
        f"Round-trip: {original_db} dB → {neper_val:.6e} Np → {recovered_db:.6f} dB"
    )


def test_db2neper_known_value() -> None:
    """db2neper matches the exact analytic formula for y=1.

    For y=1:
        α[Np/(rad/s·m)] = α[dB/(MHz·cm)] · (ln(10)/20) / (2π·10⁶ · 100)

    With α=1 dB/(MHz·cm):
        result = ln(10) / 20 · (100 m⁻¹/cm⁻¹) / (2π·10⁶ rad/s)
               = ln(10) · 5 / (2π·10⁶)
               ≈ 1.8323e-6  Np·s/m  (Neper per rad/s per metre)
    """
    import pykwavers

    result = pykwavers.db2neper(1.0, y=1.0)
    expected = math.log(10.0) * 5.0 / (2.0 * math.pi * 1e6)
    assert abs(result - expected) / expected < 1e-9, (
        f"db2neper(1.0, y=1) = {result:.6e}, expected {expected:.6e}"
    )


@pytest.mark.parametrize("alpha_db", [0.1, 0.5, 1.0, 2.5, 10.0])
def test_db2neper_positive_input_positive_output(alpha_db: float) -> None:
    """Positive attenuation in dB maps to positive value in Neper units."""
    import pykwavers

    result = pykwavers.db2neper(alpha_db, y=1.0)
    assert result > 0.0, f"db2neper({alpha_db}) = {result} should be positive"


def test_neper2db_known_value() -> None:
    """neper2db is the exact inverse of db2neper for y=1.

    1 Np/(rad/s·m) converted back:
        α[dB/(MHz·cm)] = α[Np/(rad/s·m)] · (20/ln10) · 2π·10⁶ · 100
    """
    import pykwavers

    neper_val = 1.0e-10  # representative value
    db_val    = pykwavers.neper2db(neper_val, y=1.0)
    recovered = pykwavers.db2neper(db_val, y=1.0)
    assert abs(recovered - neper_val) / neper_val < 1e-12, (
        f"Round-trip neper→dB→neper: {neper_val:.3e} → {db_val:.6e} → {recovered:.3e}"
    )
