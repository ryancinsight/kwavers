//! Quantum Optics Module — Einstein Coefficients and Radiative Transition Rates
//!
//! # Overview
//!
//! This module provides quantum optical emission models needed for complete
//! photon production analysis in extreme light-matter environments such as
//! single-bubble sonoluminescence (SBSL). The module augments the classical
//! Planck, Bremsstrahlung, and Frank-Tamm (Cherenkov) treatments with:
//!
//! 1. **Einstein A/B Coefficients** — spontaneous and stimulated emission rates
//!    for two-level atomic systems (Einstein 1917).
//! 2. **Quantum Correction Assessment** — quantitative comparison of quantum
//!    vs classical emission magnitudes for given thermodynamic conditions.
//! 3. **Relativistic Debye Length** — required for quantum plasma corrections
//!    in the Gaunt factor at high densities.
//! 4. **Sommerfeld Gaunt Factor** — frequency- and temperature-dependent
//!    correction to the classical free-free bremsstrahlung emission coefficient,
//!    replacing the constant ḡ_ff = 1.2 approximation.
//!
//! # Why Classical Models Are Sufficient for SBSL
//!
//! ## Relativistic Parameter
//!
//! At SBSL collapse temperatures T ≈ 10,000–30,000 K:
//! ```text
//! kT / (m_e c²) = 0.86–2.58 eV / 511,000 eV ≈ 2×10⁻⁶ – 5×10⁻⁶
//! ```
//! This is 5–6 orders of magnitude below the relativistic threshold. The
//! relativistic Bethe-Heitler bremsstrahlung correction is O(kT/m_e c²)
//! and is negligibly small.
//!
//! ## Lamb Shift
//!
//! The radiative correction to hydrogen 2s₁/₂ energy (Lamb 1947):
//! ```text
//! ΔE_Lamb ≈ 4.37 µeV
//! kT_{SBSL} ≈ 0.86 eV     (at 10 000 K)
//! ΔE_Lamb / kT ≈ 5 × 10⁻⁶
//! ```
//! Completely unobservable in thermal sonoluminescence spectra, where thermal
//! Doppler broadening is Δν_D / ν ≈ (2kT ln2 / mc²)^{1/2} ≈ 3 × 10⁻³.
//!
//! ## Spontaneous Emission
//!
//! The Einstein A coefficient for strong atomic transitions (oscillator strength
//! f₁₂ ~ 1) at UV wavelengths (λ ≈ 300 nm):
//! ```text
//! A₂₁ ≈ (e² ω² f₁₂) / (2πε₀ m_e c³) ≈ 6 × 10⁸ s⁻¹
//! τ_rad = 1/A₂₁ ≈ 1.7 ns
//! ```
//! During the ~100 ps sonoluminescence flash, the dimensionless number of
//! spontaneous emission events per atom is `τ_flash / τ_rad ≈ 0.06`.
//! Spontaneous emission is NOT in steady state during the flash, but it IS
//! the dominant photon production mechanism for spectral line features.
//!
//! # References
//!
//! 1. Einstein, A. (1917). Zur Quantentheorie der Strahlung. *Phys. Z.* **18**, 121–128.
//! 2. Lamb, W. E., & Retherford, R. C. (1947). Fine structure of the hydrogen
//!    atom by a microwave method. *Phys. Rev.* **72**(3), 241–243.
//! 3. Berestetskii, V. B., Lifshitz, E. M., & Pitaevskii, L. P. (1982).
//!    *Quantum Electrodynamics*. 2nd ed. Pergamon. §98 (relativistic bremsstrahlung).
//! 4. Sobelman, I. I. (1992). *Atomic Spectra and Radiative Transitions*. 2nd ed.
//!    Springer. §3.4 (Einstein coefficients), §11 (Gaunt factors).
//! 5. Sutherland, R. A. (1998). Accurate free-free Gaunt factors for astrophysical
//!    plasmas. *J. Quant. Spectrosc. Radiat. Transfer* **60**(6), 1010–1030.
//! 6. Rybicki, G. B., & Lightman, A. P. (1979). *Radiative Processes in
//!    Astrophysics*. Wiley. §5.1 (bremsstrahlung), §1.6 (Einstein coefficients).
//! 7. Brenner, M. P., Hilgenfeldt, S., & Lohse, D. (2002). Single-bubble
//!    sonoluminescence. *Rev. Mod. Phys.* **74**(2), 425–484.

use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────────

/// Boltzmann constant [J/K]
const KB: f64 = 1.380_649e-23;
/// Reduced Planck constant ħ [J·s]
const HBAR: f64 = 1.054_571_817e-34;
/// Speed of light [m/s]
const C: f64 = 2.997_924_58e8;
/// Elementary charge [C]
const E_CHARGE: f64 = 1.602_176_634e-19;
/// Electron mass [kg]
const M_E: f64 = 9.109_383_701_5e-31;
/// Vacuum permittivity [F/m]
const EPS0: f64 = 8.854_187_812_8e-12;
/// Fine structure constant α = e²/(4πε₀ħc) ≈ 1/137 (dimensionless)
/// Used in QED radiative corrections; reserved for future Lamb shift refinements.
#[allow(dead_code)]
const ALPHA_FINE: f64 = 7.297_352_569_3e-3;
/// Lamb shift for hydrogen 2s₁/₂ [eV] (Lamb & Retherford 1947)
const LAMB_SHIFT_HYDROGEN_EV: f64 = 4.374e-6;

// ─── Einstein Coefficients ────────────────────────────────────────────────────

/// Einstein A and B coefficients for a two-level atomic transition.
///
/// ## Definition (Einstein 1917)
///
/// For a transition between level 2 (upper) and level 1 (lower) with
/// angular frequency ω₂₁ = (E₂ − E₁)/ħ:
///
/// ```text
/// A₂₁ = ω₂₁³ |d₁₂|² / (3π ε₀ ħ c³)    [s⁻¹]   (spontaneous emission)
/// B₁₂ = π |d₁₂|² / (3 ε₀ ħ²)           [m³/J/s²] (absorption)
/// B₂₁ = (g₁/g₂) B₁₂                            (stimulated emission)
/// ```
///
/// **Relation**: `A₂₁ = (ħ ω₂₁³)/(π² c³) · B₂₁` (Einstein relation).
///
/// ## Oscillator Strength Formulation
///
/// Often more convenient to express via the oscillator strength f₁₂:
/// ```text
/// |d₁₂|² = (3 ħ e² f₁₂) / (2 m_e ω₂₁)
/// A₂₁     = (e² ω₂₁² f₁₂) / (2π ε₀ m_e c³)
/// ```
///
/// ## References
///
/// - Einstein (1917) Phys. Z. 18:121.
/// - Sobelman (1992) §3.4.
/// - Rybicki & Lightman (1979) §1.6.
#[derive(Debug, Clone)]
pub struct EinsteinCoefficients {
    /// Spontaneous emission rate A₂₁ [s⁻¹]
    pub a21: f64,
    /// Stimulated emission cross-section × speed of light B₂₁ [m³/J/s²]
    pub b21: f64,
    /// Absorption cross-section × speed of light B₁₂ [m³/J/s²]
    pub b12: f64,
    /// Transition angular frequency ω₂₁ [rad/s]
    pub omega21: f64,
    /// Transition dipole moment |d₁₂|² [C²·m²]
    pub dipole_moment_sq: f64,
}

impl EinsteinCoefficients {
    /// Compute Einstein coefficients from transition frequency and oscillator strength.
    ///
    /// ## Algorithm (Rybicki & Lightman 1979, §1.6; Sobelman 1992 §3.4)
    ///
    /// The oscillator strength f₁₂ is defined for the upward transition 1→2.
    /// The relation to the spontaneous emission rate A₂₁ requires the
    /// degeneracy ratio g₁/g₂ because f₁₂ is a weighted average over lower-level
    /// substates whereas A₂₁ is a rate from a specific upper level (Sobelman 1992 Eq. 3.4.3):
    ///
    /// ```text
    /// |d₁₂|² = 3ħ e² f₁₂ / (2 m_e ω₂₁)
    /// A₂₁    = (g₁/g₂) · e² ω₂₁² f₁₂ / (2π ε₀ m_e c³)
    /// B₂₁    = π |d₁₂|² / (3 ε₀ ħ²) = (π e² f₁₂) / (2 ε₀ m_e ω₂₁ ħ)
    /// B₁₂    = (g₂/g₁) B₂₁
    /// ```
    ///
    /// **Verification (hydrogen Lyman-α)**:
    /// λ = 121.567 nm, ω₂₁ = 1.549×10¹⁶ rad/s, f₁₂ = 0.4162, g₁ = 2, g₂ = 6:
    /// A₂₁ = (2/6) × (e² ω² f) / (2π ε₀ m_e c³) = 6.265×10⁸ s⁻¹  (NIST, Wiese & Fuhr 2009)
    ///
    /// ## Arguments
    ///
    /// * `omega21`    — transition frequency ω = 2πc/λ [rad/s]
    /// * `f12`        — oscillator strength (dimensionless, 0 < f ≤ 1 for typical transitions)
    /// * `g1`, `g2`   — statistical weights of lower (g₁) and upper (g₂) levels
    #[must_use]
    pub fn from_oscillator_strength(omega21: f64, f12: f64, g1: f64, g2: f64) -> Self {
        // Dipole moment squared [C² m²]  (Rybicki & Lightman 1979 Eq. 1.68)
        let dipole_sq = 3.0 * HBAR * E_CHARGE * E_CHARGE * f12 / (2.0 * M_E * omega21);

        // A coefficient: includes degeneracy ratio g₁/g₂ (Sobelman 1992 §3.4, Eq. 3.4.3)
        // A₂₁ = (g₁/g₂) · e² ω₂₁² f₁₂ / (2π ε₀ m_e c³)
        let a21 = (E_CHARGE * E_CHARGE * omega21 * omega21 * f12 * g1)
            / (2.0 * PI * EPS0 * M_E * C * C * C * g2.max(f64::EPSILON));

        // B coefficient (stimulated emission) from dipole matrix element
        let b21 = (PI * dipole_sq) / (3.0 * EPS0 * HBAR * HBAR);

        // B coefficient (absorption) from detailed balance: B₁₂ = (g₂/g₁) B₂₁
        let b12 = b21 * g2 / g1.max(f64::EPSILON);

        Self {
            a21,
            b21,
            b12,
            omega21,
            dipole_moment_sq: dipole_sq,
        }
    }

    /// Radiative lifetime τ = 1/A₂₁ [s].
    ///
    /// For hydrogen Lyman-α (λ = 121.6 nm, f₁₂ = 0.4162, ω ≈ 1.55e16 rad/s):
    /// τ_rad ≈ 1.6 ns (Wiese & Fuhr 2009).
    #[must_use]
    pub fn radiative_lifetime(&self) -> f64 {
        if self.a21 < f64::EPSILON {
            f64::INFINITY
        } else {
            1.0 / self.a21
        }
    }

    /// Fraction of excited atoms that spontaneously emit during a flash of duration `dt` [s].
    ///
    /// `f_emit = 1 − exp(−A₂₁ · dt)` ≈ `A₂₁ · dt` for short flashes.
    #[must_use]
    pub fn flash_emission_fraction(&self, dt: f64) -> f64 {
        1.0 - (-self.a21 * dt).exp()
    }
}

// ─── Sommerfeld-Bethe Gaunt Factor ────────────────────────────────────────────

/// Modified Bessel function of the second kind, order zero: K₀(x) for x > 0.
///
/// ## Algorithm (Abramowitz & Stegun §9.8.5–9.8.6; Numerical Recipes §6.6)
///
/// **For 0 < x ≤ 2** (A&S 9.8.1 + 9.8.5):
/// ```text
/// I₀(x) = Σ_{n=0}^{6} d_n · (x/3.75)^{2n}           [A&S 9.8.1]
/// K₀(x) = −ln(x/2)·I₀(x) + Σ_{n=0}^{6} c_n·(x/2)^{2n}  [A&S 9.8.5]
/// ```
///
/// **For x > 2** (A&S 9.8.6):
/// ```text
/// K₀(x) = exp(−x)/√x · Σ_{n=0}^{6} p_n·(2/x)^n
/// ```
///
/// Accuracy: |ε| < 1.6×10⁻⁷ for both ranges.
///
/// ## Reference
/// Abramowitz, M., & Stegun, I. A. (1972). *Handbook of Mathematical Functions*,
/// §9.8.1, 9.8.5–9.8.6. National Bureau of Standards.
fn bessel_k0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        // I₀(x) via polynomial in t₁ = (x/3.75)²  (A&S 9.8.1)
        let t1 = (x / 3.75) * (x / 3.75);
        let i0 = 1.0
            + t1 * (3.515_632_9
                + t1 * (3.089_942_4
                    + t1 * (1.206_749_2
                        + t1 * (0.265_973_2 + t1 * (0.036_076_8 + t1 * 0.004_581_3)))));
        // K₀ correction polynomial in t₂ = (x/2)²  (A&S 9.8.5)
        let t2 = (x * 0.5) * (x * 0.5);
        let corr = -0.577_215_66
            + t2 * (0.422_784_20
                + t2 * (0.230_697_56
                    + t2 * (0.034_885_90
                        + t2 * (0.002_626_98 + t2 * (0.000_107_50 + t2 * 7.4e-6)))));
        corr - (x * 0.5).ln() * i0
    } else {
        // Asymptotic expansion in t = 2/x  (A&S 9.8.6)
        let t = 2.0 / x;
        let poly = 1.253_314_14
            + t * (-0.078_323_58
                + t * (0.021_895_68
                    + t * (-0.010_624_46
                        + t * (0.005_878_72 + t * (-0.002_515_40 + t * 0.000_532_08)))));
        (-x).exp() / x.sqrt() * poly
    }
}

/// Frequency- and temperature-dependent free-free Gaunt factor.
///
/// ## Algorithm (Sutherland 1998, Eq. 11; Rybicki & Lightman 1979 §5.1 Eq. 5.14b)
///
/// The thermally-averaged free-free Gaunt factor for a non-degenerate hydrogen-like
/// plasma, integrated over the Maxwellian velocity distribution:
///
/// ```text
/// ḡ_ff(ν, T) = (√3 / π) · exp(u/2) · K₀(u/2)
/// ```
///
/// where:
/// ```text
/// u = hν / kT    [dimensionless photon energy]
/// K₀             [modified Bessel function of the second kind, order 0]
/// ```
///
/// This formula (Rybicki & Lightman 1979 Eq. 5.14b) is derived by integrating the
/// differential free-free cross section over a Maxwell-Boltzmann distribution. It:
/// - Is **always positive**: K₀(x) > 0 and exp(x) > 0 for all x > 0
/// - Reduces to the Born approximation `(√3/π)[ln(2kT/hν) − γ_EM]` for u ≪ 1
/// - Monotonically **decreases with frequency** at fixed T (ḡ_ff falls at high u)
/// - Gives ḡ_ff ≈ 0.48–0.66 for SBSL visible/UV conditions (T=10 kK, λ=200–800 nm)
/// - The constant approximation ḡ_ff = 1.2 (Rybicki & Lightman) overestimates
///   by ~2× in the SBSL UV regime but is useful as a rough upper bound.
///
/// **Note on the Born (logarithmic) form**: The formula
/// `(√3/π) · [ln(χ) + C_E]` is only valid for u ≪ 1 (Rayleigh-Jeans regime)
/// and gives *negative* Gaunt factors for u > ~1.5 — it must **not** be used
/// for SBSL UV conditions where u = hν/kT ≈ 2–7.
///
/// ## Arguments
///
/// * `freq_hz`       — photon frequency ν [Hz]
/// * `temperature_k` — electron temperature T [K]
/// * `_z`            — ionic charge number (Z; cancels in the thermally-averaged
///   ratio, retained for API compatibility)
///
/// ## References
///
/// - Rybicki, G. B., & Lightman, A. P. (1979) §5.1 Eq. 5.14b.
/// - Sutherland, R. A. (1998) J. Quant. Spectrosc. Radiat. Transfer 60(6):1010,
///   Eq. 11 (general thermally-averaged form).
/// - Karzas, W. J., & Latter, R. (1961) Astrophys. J. Suppl. 6:167.
#[must_use]
pub fn gaunt_factor_ff(freq_hz: f64, temperature_k: f64, _z: f64) -> f64 {
    const H_PLANCK: f64 = 6.626_070_15e-34;

    if freq_hz < 1e6 || temperature_k < 1.0 {
        return 1.0; // fallback for unphysical inputs
    }

    // u = hν/kT (dimensionless photon energy)
    let u = H_PLANCK * freq_hz / (KB * temperature_k);

    // Sutherland (1998) / Rybicki-Lightman thermally-averaged formula:
    // ḡ_ff = (√3/π) · exp(u/2) · K₀(u/2)
    let x = 0.5 * u;
    let g_ff = (3.0_f64.sqrt() / PI) * x.exp() * bessel_k0(x);

    // Clamp to physically reasonable range; formula is accurate for all u > 0
    g_ff.clamp(0.1, 10.0)
}

// ─── Relativistic Correction Assessment ──────────────────────────────────────

/// Compute the relativistic parameter kT/(m_e c²) for plasma electrons.
///
/// This dimensionless ratio determines whether relativistic corrections
/// to bremsstrahlung (Bethe-Heitler formula) are needed:
/// - `kT/(m_e c²) ≪ 0.01`: classical bremsstrahlung valid to < 1%
/// - `kT/(m_e c²) ≈ 0.1`:  relativistic corrections ~10%
/// - `kT/(m_e c²) > 1`:    fully relativistic, Bethe-Heitler required
///
/// For SBSL (T ≈ 10,000–30,000 K): `kT/(m_e c²) ≈ 2–6 × 10⁻⁶`
/// → Classical bremsstrahlung accurate to < 10 parts per million.
///
/// ## Reference
/// Berestetskii, Lifshitz & Pitaevskii (1982) §98.
#[must_use]
pub fn relativistic_parameter(temperature_k: f64) -> f64 {
    KB * temperature_k / (M_E * C * C)
}

/// Compute the Lamb shift in eV for hydrogen-like atoms with atomic number Z.
///
/// ## Algorithm (Berestetskii, Lifshitz & Pitaevskii 1982, §104)
///
/// For the 2s₁/₂ level of a hydrogen-like atom with nuclear charge Z:
/// ```text
/// ΔE_Lamb ≈ (α⁵ m_e c² Z⁴ / (12π)) · [ln(α⁻² Z⁻²) + C_QED]
/// ```
///
/// For Z = 1 (hydrogen): ΔE_Lamb ≈ 4.37 µeV (Lamb 1947 measurement).
/// Scales as Z⁴ for other hydrogen-like ions.
///
/// The Lamb shift is negligible compared to thermal broadening at SBSL
/// temperatures: `ΔE_Lamb / kT ≈ 4.37e-6 eV / 0.86 eV ≈ 5 × 10⁻⁶`.
///
/// ## Arguments
///
/// * `z` — atomic number (1 for hydrogen)
///
/// ## Returns
///
/// Lamb shift in eV [eV]
///
/// ## References
///
/// - Lamb, W. E., & Retherford, R. C. (1947) Phys. Rev. 72(3):241.
/// - Berestetskii, Lifshitz & Pitaevskii (1982) §104.
#[must_use]
pub fn lamb_shift_ev(z: f64) -> f64 {
    // Scaling: ΔE_Lamb(Z) ≈ ΔE_Lamb(H) × Z⁴ (leading term)
    LAMB_SHIFT_HYDROGEN_EV * z.powi(4)
}

/// Assess whether quantum corrections are physically significant for given conditions.
///
/// Returns a `QuantumCorrectionAssessment` with magnitude estimates for each
/// quantum effect relative to the classical emission intensity.
#[derive(Debug, Clone)]
pub struct QuantumCorrectionAssessment {
    /// Relativistic parameter kT/(m_e c²) — should be ≪ 0.01
    pub relativistic_parameter: f64,
    /// Lamb shift / kT ratio — should be ≪ 1
    pub lamb_shift_ratio: f64,
    /// Flash duration / radiative lifetime ratio — determines spontaneous emission significance
    pub flash_to_lifetime_ratio: f64,
    /// Classical bremsstrahlung accuracy (%): 100 × (1 − relativistic_correction)
    pub classical_accuracy_pct: f64,
}

impl QuantumCorrectionAssessment {
    /// Assess quantum corrections for given temperature and flash duration.
    ///
    /// ## Arguments
    ///
    /// * `temperature_k`  — plasma temperature [K]
    /// * `flash_duration_s` — sonoluminescence pulse duration [s]
    /// * `transition_omega` — dominant atomic transition frequency [rad/s]
    /// * `oscillator_strength` — oscillator strength f₁₂ of dominant transition
    #[must_use]
    pub fn assess(
        temperature_k: f64,
        flash_duration_s: f64,
        transition_omega: f64,
        oscillator_strength: f64,
    ) -> Self {
        let rel_param = relativistic_parameter(temperature_k);
        let k_t_ev = KB * temperature_k / E_CHARGE;
        let lamb_ratio = lamb_shift_ev(1.0) / k_t_ev;

        // Einstein A coefficient for the specified transition
        let einstein = EinsteinCoefficients::from_oscillator_strength(
            transition_omega,
            oscillator_strength,
            1.0,
            1.0,
        );
        let flash_ratio = flash_duration_s * einstein.a21;

        // Classical bremsstrahlung accuracy (first-order relativistic correction ≈ O(kT/m_e c²))
        let classical_accuracy = (1.0 - rel_param).max(0.0) * 100.0;

        Self {
            relativistic_parameter: rel_param,
            lamb_shift_ratio: lamb_ratio,
            flash_to_lifetime_ratio: flash_ratio,
            classical_accuracy_pct: classical_accuracy,
        }
    }

    /// True if classical bremsstrahlung is accurate to within 0.1%
    #[must_use]
    pub fn classical_bremsstrahlung_adequate(&self) -> bool {
        self.relativistic_parameter < 1e-3
    }

    /// True if Lamb shift can be neglected (< 1% of kT)
    #[must_use]
    pub fn lamb_shift_negligible(&self) -> bool {
        self.lamb_shift_ratio < 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Einstein Coefficients ─────────────────────────────────────────────────

    /// Lyman-α (H): λ = 121.567 nm, f₁₂ = 0.4162, g₁=2, g₂=6.
    /// Reference: Wiese & Fuhr (2009) J. Phys. Chem. Ref. Data 38(3):565.
    /// A₂₁ = 6.265 × 10⁸ s⁻¹ (NIST), τ = 1.596 ns.
    #[test]
    fn test_einstein_a21_hydrogen_lyman_alpha() {
        let c = 2.997924e8_f64;
        let lambda = 121.567e-9; // m
        let omega21 = 2.0 * std::f64::consts::PI * c / lambda;
        let f12 = 0.4162;
        let coeff = EinsteinCoefficients::from_oscillator_strength(omega21, f12, 2.0, 6.0);
        // NIST reference: A₂₁ = 6.265e8 s⁻¹
        let expected = 6.265e8;
        let rel_err = (coeff.a21 - expected).abs() / expected;
        assert!(
            rel_err < 0.05, // 5% tolerance (Born approximation accuracy)
            "Lyman-α A₂₁: got {:.4e}, expected {:.4e} (err {:.1}%)",
            coeff.a21,
            expected,
            100.0 * rel_err
        );
    }

    /// Radiative lifetime for hydrogen Lyman-α must be ~1.6 ns.
    #[test]
    fn test_radiative_lifetime_lyman_alpha() {
        let c = 2.997924e8_f64;
        let lambda = 121.567e-9;
        let omega21 = 2.0 * std::f64::consts::PI * c / lambda;
        let coeff = EinsteinCoefficients::from_oscillator_strength(omega21, 0.4162, 2.0, 6.0);
        let tau = coeff.radiative_lifetime();
        // τ = 1/A₂₁ ≈ 1.596 ns (NIST)
        assert!(
            tau > 1.0e-9 && tau < 3.0e-9,
            "Lyman-α lifetime must be 1-3 ns, got {tau:.3e} s"
        );
    }

    /// Einstein B₁₂ > B₂₁ for g₁ < g₂ (correct degeneracy scaling).
    #[test]
    fn test_einstein_b_degeneracy_relation() {
        let c = 2.997924e8_f64;
        let omega = 2.0 * std::f64::consts::PI * c / 400e-9;
        let coeff = EinsteinCoefficients::from_oscillator_strength(omega, 0.5, 1.0, 3.0);
        // B₁₂ = (g₂/g₁) B₂₁ = 3 × B₂₁
        let ratio = coeff.b12 / coeff.b21;
        assert!(
            (ratio - 3.0).abs() < 1e-13,
            "B₁₂/B₂₁ = g₂/g₁ = 3, got {ratio:.10}"
        );
    }

    /// Flash emission fraction must be small for τ_flash ≪ τ_rad.
    #[test]
    fn test_flash_emission_fraction_sbsl() {
        let c = 2.997924e8_f64;
        let omega = 2.0 * std::f64::consts::PI * c / 300e-9;
        let coeff = EinsteinCoefficients::from_oscillator_strength(omega, 0.4, 1.0, 1.0);
        let flash_dt = 100e-12; // 100 ps
        let frac = coeff.flash_emission_fraction(flash_dt);
        // At 100 ps, fraction ≈ A₂₁ × dt ≈ A₂₁ × 1e-10
        assert!(
            frac > 0.0 && frac < 0.3,
            "Flash emission fraction must be < 30% for short flash"
        );
        assert!(
            frac < coeff.a21 * flash_dt * 1.05,
            "Flash fraction must approach A₂₁·dt for short flash"
        );
    }

    // ── Gaunt Factor ──────────────────────────────────────────────────────────

    /// Gaunt factor at SBSL conditions (T=10000K, λ=400nm, Z=1).
    ///
    /// At u = hν/kT = 3.1 eV / 0.86 eV ≈ 3.6, the Sutherland formula gives
    /// ḡ_ff = (√3/π)·exp(1.8)·K₀(1.8) ≈ 0.48–0.50, which lies in (0.3, 1.5).
    #[test]
    fn test_gaunt_factor_sbsl_conditions() {
        let c = 2.997924e8_f64;
        let nu = c / 400e-9;
        let g_ff = gaunt_factor_ff(nu, 10_000.0, 1.0);
        assert!(
            (0.3..=1.5).contains(&g_ff),
            "Gaunt factor at SBSL conditions must be in [0.3, 1.5], got {g_ff:.4}"
        );
    }

    /// Gaunt factor must decrease with increasing frequency (blue side).
    #[test]
    fn test_gaunt_factor_frequency_dependence() {
        let c = 2.997924e8_f64;
        let g_red = gaunt_factor_ff(c / 800e-9, 10_000.0, 1.0);
        let g_uv = gaunt_factor_ff(c / 200e-9, 10_000.0, 1.0);
        // In the Born approximation, ḡ_ff decreases at high u = hν/kT
        assert!(
            g_red > g_uv,
            "Gaunt factor at 800 nm ({g_red:.4}) must exceed that at 200 nm ({g_uv:.4})"
        );
    }

    // ── Relativistic / Lamb shift assessment ─────────────────────────────────

    /// Relativistic parameter at SBSL temperature must be << 1e-3.
    #[test]
    fn test_relativistic_parameter_sbsl() {
        let rel = relativistic_parameter(15_000.0);
        assert!(
            rel < 1e-3,
            "At 15 000 K, relativistic parameter = {rel:.3e} must be << 1e-3"
        );
    }

    /// Lamb shift ratio must be << 0.01 at SBSL temperatures.
    #[test]
    fn test_lamb_shift_negligible_at_sbsl_temperature() {
        let k_t_ev = KB * 10_000.0 / E_CHARGE;
        let ratio = lamb_shift_ev(1.0) / k_t_ev;
        assert!(
            ratio < 1e-4,
            "Lamb shift / kT at 10 000 K = {ratio:.3e} must be < 1e-4"
        );
    }

    /// Classical bremsstrahlung must be adequate (> 99.99%) for SBSL.
    #[test]
    fn test_classical_bremsstrahlung_adequate_sbsl() {
        let c = 2.997924e8_f64;
        let omega_uv = 2.0 * std::f64::consts::PI * c / 300e-9;
        let assessment = QuantumCorrectionAssessment::assess(10_000.0, 100e-12, omega_uv, 0.4);
        assert!(
            assessment.classical_bremsstrahlung_adequate(),
            "Classical bremsstrahlung must be adequate at 10 000 K"
        );
        assert!(
            assessment.lamb_shift_negligible(),
            "Lamb shift must be negligible at 10 000 K"
        );
        assert!(
            assessment.classical_accuracy_pct > 99.99,
            "Classical accuracy at 10 000 K: {:.6}%",
            assessment.classical_accuracy_pct
        );
    }
}
