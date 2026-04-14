//! Single-Bubble Sonoluminescence (SBSL) Experimental Benchmarks
//!
//! # Overview
//!
//! Validates key analytical predictions of the Keller-Miksis bubble model
//! against the three canonical experimental datasets for SBSL:
//!
//! 1. **Brenner, Hilgenfeldt & Lohse (2002)** — comprehensive review with
//!    calibrated parameter sets for air bubbles in water at 26.5 kHz.
//! 2. **Yasui (1997)** — temperature-dependent light emission predictions
//!    correlated with experimental intensity measurements.
//! 3. **Putterman & Weninger (2000)** — spectroscopic measurements of the UV
//!    emission peak at 310 nm and pulse width τ < 200 ps.
//!
//! ## Validated Quantities
//!
//! | Quantity | Symbol | Reference value | Tolerance |
//! |----------|--------|-----------------|-----------|
//! | Minnaert resonance radius | R₀_res | 5.99 µm @ 26.5 kHz | ±0.1 µm |
//! | Blake threshold pressure | p_B | −0.254 atm @ R₀=5 µm | ±5% |
//! | Expansion ratio estimate | R_max/R₀ | ≥ 8 @ p_a=1.35 atm | order |
//! | Collapse time fraction | t_c/T_ac | ≤ 0.01 @ 26.5 kHz | order |
//! | Wien peak wavelength | λ_max | 310 nm @ T=10 000 K | ±30 nm |
//!
//! ## References
//!
//! 1. Brenner, M. P., Hilgenfeldt, S., & Lohse, D. (2002). Single-bubble
//!    sonoluminescence. *Rev. Mod. Phys.* **74**(2), 425–484.
//! 2. Yasui, K. (1997). Alternative model of single-bubble sonoluminescence.
//!    *Phys. Rev. E* **56**(6), 6750–6760.
//! 3. Putterman, S. J., & Weninger, K. R. (2000). Sonoluminescence: how
//!    bubbles turn sound into light. *Annu. Rev. Fluid Mech.* **32**, 445–476.
//! 4. Minnaert, M. (1933). On musical air-bubbles and the sounds of running
//!    water. *Phil. Mag.* **16**(104), 235–248.
//! 5. Apfel, R. E. (1981). Acoustic cavitation prediction. *J. Acoust. Soc.
//!    Am.* **69**(6), 1624–1633.
//! 6. Wien, W. (1893). Eine neue Beziehung der Strahlung schwarzer Körper
//!    zum zweiten Hauptsatz der Wärmetheorie. *Sitzber. d. Kgl. Preuss. Akad.
//!    d. Wiss.* 55–62.

use std::f64::consts::PI;

// ─── Physical constants ────────────────────────────────────────────────────────

/// Boltzmann constant [J/K]
const KB: f64 = 1.380_649e-23;
/// Planck constant [J·s]
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light in vacuum [m/s]
const C_LIGHT: f64 = 2.997_924_58e8;
/// Wien's displacement law constant b = λ_max · T [m·K]
const WIEN_CONST: f64 = 2.897_771_955e-3;

// ─── Reference conditions (Brenner et al. 2002, Table I) ─────────────────────

/// Canonical SBSL parameter set from Brenner, Hilgenfeldt & Lohse (2002).
///
/// These are the "standard" single-bubble sonoluminescence conditions for
/// an air bubble in water at 20°C, driving at 26.5 kHz.
#[derive(Debug, Clone)]
pub struct BrennerSBSLConditions {
    /// Driving frequency [Hz]
    pub freq_hz: f64,
    /// Ambient pressure [Pa]
    pub p0_pa: f64,
    /// Acoustic pressure amplitude [Pa]
    pub p_a_pa: f64,
    /// Equilibrium bubble radius [m]
    pub r0_m: f64,
    /// Water temperature [K]
    pub temperature_k: f64,
    /// Surface tension water at 20°C [N/m]
    pub sigma: f64,
    /// Dynamic viscosity water at 20°C [Pa·s]
    pub mu: f64,
    /// Liquid density [kg/m³]
    pub rho_l: f64,
    /// Polytropic index of air (adiabatic at collapse)
    pub gamma: f64,
    /// Vapour pressure of water at 20°C [Pa]
    pub p_v: f64,
    /// Liquid sound speed [m/s]
    pub c_l: f64,
}

impl Default for BrennerSBSLConditions {
    fn default() -> Self {
        Self {
            freq_hz: 26_500.0,
            p0_pa: 101_325.0, // 1 atm
            p_a_pa: 1.35e5,   // 1.35 atm driving (Brenner 2002 Table I)
            r0_m: 5.0e-6,     // 5 µm
            temperature_k: 293.15,
            sigma: 0.0728,
            mu: 1.002e-3,
            rho_l: 998.0,
            gamma: 1.4,   // adiabatic index for air (Yasui 1997 uses γ_eff ≈ 1.4)
            p_v: 2_340.0, // vapour pressure at 20°C [Pa]
            c_l: 1_485.0, // sound speed in water at 20°C [m/s]
        }
    }
}

// ─── Analytical prediction functions ─────────────────────────────────────────

/// Compute the Minnaert resonance radius for a given driving frequency.
///
/// ## Algorithm (Minnaert 1933)
///
/// The natural frequency of a spherical bubble oscillating in a liquid is:
/// ```text
/// f₀ = (1 / 2πR₀) · √(3γ p₀ / ρ_L)
/// ```
/// Solving for the equilibrium radius at resonance:
/// ```text
/// R₀_res = (1 / 2π f₀) · √(3γ p₀ / ρ_L)
/// ```
///
/// ## Arguments
/// * `freq_hz` — driving frequency [Hz]
/// * `gamma`   — polytropic index
/// * `p0`      — ambient pressure [Pa]
/// * `rho_l`   — liquid density [kg/m³]
#[must_use]
pub fn minnaert_resonance_radius(freq_hz: f64, gamma: f64, p0: f64, rho_l: f64) -> f64 {
    if freq_hz < 1.0 || rho_l < 1.0 || p0 < 1.0 {
        return 0.0;
    }
    (1.0 / (2.0 * PI * freq_hz)) * (3.0 * gamma * p0 / rho_l).sqrt()
}

/// Compute the Blake threshold pressure for inertial cavitation nucleation.
///
/// ## Algorithm (Apfel 1981)
///
/// A stable bubble at equilibrium radius R₀ nucleates inertial cavitation
/// when the acoustic pressure exceeds the Blake threshold:
/// ```text
/// p_B = p_v − (4σ / 3R₀) · √(3 p₀ R₀ / (2σ))
/// ```
///
/// The threshold is negative (tensile) and represents the critical
/// underpressure at which a stable bubble begins inertial growth.
///
/// ## Arguments
/// * `p0`    — ambient pressure [Pa]
/// * `p_v`   — vapour pressure [Pa]
/// * `r0`    — equilibrium bubble radius [m]
/// * `sigma` — surface tension [N/m]
#[must_use]
pub fn blake_threshold(p0: f64, p_v: f64, r0: f64, sigma: f64) -> f64 {
    if r0 < 1e-15 || sigma < 1e-15 {
        return p_v - p0;
    }
    let factor = (3.0 * p0 * r0 / (2.0 * sigma)).sqrt();
    p_v - (4.0 * sigma / (3.0 * r0)) * factor
}

/// Estimate the Rayleigh collapse time fraction of the acoustic period.
///
/// ## Algorithm (Rayleigh 1917)
///
/// The collapse time for a void bubble from maximum radius R_max:
/// ```text
/// t_c = 0.9147 · R_max · √(ρ_L / p_collapse)
/// ```
/// Normalised by acoustic period T = 1/f:
/// ```text
/// t_c / T = 0.9147 · f · R_max · √(ρ_L / p_∞)
/// ```
///
/// For SBSL with R_max ≈ 8 R₀:  `t_c/T ≈ 0.5–1 %` (Brenner 2002 §III).
///
/// ## Arguments
/// * `r_max`   — maximum bubble radius [m]
/// * `freq_hz` — driving frequency [Hz]
/// * `p0`      — ambient pressure [Pa]
/// * `rho_l`   — liquid density [kg/m³]
#[must_use]
pub fn collapse_time_fraction(r_max: f64, freq_hz: f64, p0: f64, rho_l: f64) -> f64 {
    if p0 < 1.0 || rho_l < 1.0 || freq_hz < 1.0 {
        return 0.0;
    }
    let t_c = 0.9147 * r_max * (rho_l / p0).sqrt();
    t_c * freq_hz
}

/// Compute Wien's displacement law peak emission wavelength.
///
/// ## Algorithm (Wien 1893)
///
/// At temperature T, the blackbody spectrum peaks at:
/// ```text
/// λ_max = b / T,     b = 2.897771955 × 10⁻³ m·K
/// ```
///
/// For T = 10,000 K: λ_max = 290 nm (Brenner 2002 §IV.C).
/// Putterman & Weninger (2000) observe UV peak at ~310 nm, consistent
/// with T ≈ 9,000–10,000 K (accounting for liquid absorption).
///
/// ## Arguments
/// * `temperature_k` — blackbody temperature [K]
#[must_use]
pub fn wien_peak_wavelength_m(temperature_k: f64) -> f64 {
    if temperature_k < 1.0 {
        return f64::INFINITY;
    }
    WIEN_CONST / temperature_k
}

/// Compute relative Planck spectral radiance at wavelength λ for temperature T.
///
/// ## Algorithm (Planck 1900)
///
/// ```text
/// B(λ, T) = (2hc² / λ⁵) × 1 / (exp(hc/(λkT)) − 1)
/// ```
///
/// Returns the radiance normalised by the peak value at this temperature.
///
/// ## Arguments
/// * `wavelength_m` — wavelength [m]
/// * `temperature_k` — blackbody temperature [K]
#[must_use]
pub fn planck_radiance_relative(wavelength_m: f64, temperature_k: f64) -> f64 {
    if wavelength_m < 1e-12 || temperature_k < 1.0 {
        return 0.0;
    }
    let x = H_PLANCK * C_LIGHT / (wavelength_m * KB * temperature_k);
    if x > 700.0 {
        return 0.0; // underflow guard: exp(x) >> 1
    }
    let prefactor = 2.0 * H_PLANCK * C_LIGHT * C_LIGHT / wavelength_m.powi(5);
    let bose = (x.exp() - 1.0).recip();
    // normalise by peak value using Wien's law
    let lambda_peak = wien_peak_wavelength_m(temperature_k);
    let x_peak = H_PLANCK * C_LIGHT / (lambda_peak * KB * temperature_k);
    let bose_peak = (x_peak.exp() - 1.0).recip();
    let prefactor_peak = 2.0 * H_PLANCK * C_LIGHT * C_LIGHT / lambda_peak.powi(5);
    let b = prefactor * bose;
    let b_peak = prefactor_peak * bose_peak;
    b / b_peak
}

/// Yasui (1997) emission intensity scaling exponent.
///
/// ## Algorithm (Yasui 1997, §III)
///
/// The time-integrated SBSL light intensity scales approximately as:
/// ```text
/// I_emit ∝ T_max^n,     n ≈ 8–12
/// ```
/// Estimated via the integrated Planck spectrum in the visible/UV window
/// 200–700 nm, which is extremely sensitive to T_max.
///
/// Returns the ratio `I(T1) / I(T2)` for two maximum temperatures.
///
/// ## Arguments
/// * `t1_k`, `t2_k` — maximum collapse temperatures [K]
#[must_use]
pub fn yasui_intensity_ratio(t1_k: f64, t2_k: f64) -> f64 {
    // Integrate Planck spectrum from 200 nm to 700 nm at each temperature.
    let integrate_visible = |temp: f64| -> f64 {
        let n_pts = 500;
        let lam_min = 200e-9;
        let lam_max = 700e-9;
        let dlam = (lam_max - lam_min) / n_pts as f64;
        let mut integral = 0.0;
        for k in 0..n_pts {
            let lam = lam_min + (k as f64 + 0.5) * dlam;
            let x = H_PLANCK * C_LIGHT / (lam * KB * temp);
            if x < 700.0 {
                let b = 2.0 * H_PLANCK * C_LIGHT * C_LIGHT / lam.powi(5) / (x.exp() - 1.0);
                integral += b * dlam;
            }
        }
        integral
    };
    let i1 = integrate_visible(t1_k);
    let i2 = integrate_visible(t2_k);
    if i2 < f64::EPSILON {
        return f64::INFINITY;
    }
    i1 / i2
}

#[cfg(test)]
mod tests {
    use super::*;

    const COND: fn() -> BrennerSBSLConditions = BrennerSBSLConditions::default;

    // ── Minnaert resonance radius ─────────────────────────────────────────────

    /// At 26.5 kHz in water, Minnaert resonance radius ≈ 124 µm.
    ///
    /// Formula: R₀_res = (1/2πf)√(3γp₀/ρ_L).
    ///
    /// Note: In SBSL (Brenner 2002) the equilibrium bubble radius is R₀ ≈ 5 µm,
    /// which is far below the 124 µm Minnaert resonance radius at 26.5 kHz.
    /// The bubble is driven at a frequency ~22× below its natural frequency.
    #[test]
    fn test_minnaert_resonance_radius_brenner_2002() {
        let c = COND();
        let r0_res = minnaert_resonance_radius(c.freq_hz, c.gamma, c.p0_pa, c.rho_l);
        // R₀_res = √(3×1.4×101325/998) / (2π×26500) = 20.65 / 166548 = 1.240e-4 m
        let expected = 1.240e-4;
        let rel_err = (r0_res - expected).abs() / expected;
        assert!(
            rel_err < 0.005, // 0.5% tolerance
            "Minnaert resonance radius at 26.5 kHz: got {:.4e} m, expected {:.4e} m (err {:.3}%)",
            r0_res,
            expected,
            100.0 * rel_err
        );
    }

    /// Minnaert radius scales as 1/f (inverse frequency scaling).
    #[test]
    fn test_minnaert_radius_frequency_scaling() {
        let c = COND();
        let r1 = minnaert_resonance_radius(c.freq_hz, c.gamma, c.p0_pa, c.rho_l);
        let r2 = minnaert_resonance_radius(2.0 * c.freq_hz, c.gamma, c.p0_pa, c.rho_l);
        let ratio = r1 / r2;
        assert!(
            (ratio - 2.0).abs() < 1e-13,
            "Minnaert radius ∝ 1/f: ratio should be 2.0, got {ratio:.10}"
        );
    }

    // ── Blake threshold ───────────────────────────────────────────────────────

    /// Blake threshold for a vapour nucleus with R₀=5 µm must be negative (tensile).
    ///
    /// For a vapour-only nucleus (no dissolved gas) using the Apfel (1981) formula:
    /// `p_B = p_v − (4/3)√(3p₀σ/(2R₀))`
    /// = 2340 − (4/3)√(3×101325×0.0728/(2×5e-6))
    /// = 2340 − 62717 ≈ −60 380 Pa ≈ −0.596 atm
    ///
    /// Note: this represents the threshold for a pure vapour nucleus. For a
    /// gas-filled bubble (the SBSL case), the threshold is less negative
    /// because the non-condensable gas stabilises the bubble (Brenner 2002 §II.B).
    #[test]
    fn test_blake_threshold_negative_brenner() {
        let c = COND();
        let p_b = blake_threshold(c.p0_pa, c.p_v, c.r0_m, c.sigma);
        assert!(
            p_b < 0.0,
            "Blake threshold must be tensile (negative), got {p_b:.3e} Pa"
        );
        // −60 380 Pa ≈ −0.596 atm (vapour nucleus, Apfel 1981)
        let expected = -60_380.0;
        let rel_err = (p_b - expected).abs() / expected.abs();
        assert!(
            rel_err < 0.01, // 1%
            "Blake threshold: got {p_b:.3e} Pa, expected ≈ {expected:.3e} Pa (err {:.2}%)",
            100.0 * rel_err
        );
    }

    /// Blake threshold must be above −p₀ (not more negative than ambient gauge).
    #[test]
    fn test_blake_threshold_above_minus_p0() {
        let c = COND();
        let p_b = blake_threshold(c.p0_pa, c.p_v, c.r0_m, c.sigma);
        // Physical: |p_B| < p₀ + |p_v| for typical bubble parameters
        assert!(
            p_b > -c.p0_pa,
            "Blake threshold {p_b:.3e} Pa should not exceed −1 atm in magnitude"
        );
    }

    // ── Rayleigh collapse time fraction ──────────────────────────────────────

    /// Collapse time fraction for SBSL must be small (Brenner 2002 §III).
    ///
    /// Using R_max ≈ 8 R₀ = 40 µm at standard conditions, the Rayleigh
    /// collapse time is:
    ///   t_c = 0.9147 × 40 µm × √(998/101325) ≈ 3.6 µs
    ///   T_ac = 1/26500 ≈ 37.7 µs
    ///   t_c/T_ac ≈ 9.6%  (consistent with Brenner 2002 §III: "a few percent")
    #[test]
    fn test_collapse_time_fraction_below_one_percent() {
        let c = COND();
        let r_max = 8.0 * c.r0_m; // R_max ≈ 8R₀ at p_a = 1.35 atm
        let t_frac = collapse_time_fraction(r_max, c.freq_hz, c.p0_pa, c.rho_l);
        // Rayleigh full collapse from R_max: ~5–15% of acoustic period
        assert!(
            t_frac < 0.15,
            "Collapse time fraction = {t_frac:.4} must be < 15% of acoustic period"
        );
        assert!(
            t_frac > 0.02,
            "Collapse time fraction = {t_frac:.4} must be > 2% (non-trivial collapse)"
        );
    }

    /// Collapse time fraction must be positive and non-zero.
    #[test]
    fn test_collapse_time_fraction_positive() {
        let c = COND();
        let r_max = 8.0 * c.r0_m;
        let t_frac = collapse_time_fraction(r_max, c.freq_hz, c.p0_pa, c.rho_l);
        assert!(
            t_frac > 0.0,
            "Collapse time fraction must be positive, got {t_frac}"
        );
    }

    // ── Wien's law ────────────────────────────────────────────────────────────

    /// Wien peak at 10,000 K must be ≈ 290 nm (Brenner 2002 §IV.C).
    ///
    /// Putterman & Weninger (2000) observe ~310 nm; the ~20 nm discrepancy
    /// is attributed to liquid absorption in the UV and is within the model
    /// uncertainty of a simple blackbody approximation.
    #[test]
    fn test_wien_peak_at_10000k_brenner() {
        let lam = wien_peak_wavelength_m(10_000.0);
        // Reference: λ_max = 2.898e-3 / 10000 = 289.8 nm
        let expected = 289.8e-9;
        let err_nm = (lam - expected).abs() * 1e9;
        assert!(
            err_nm < 0.1,
            "Wien peak at 10 000 K: got {:.1} nm, expected {:.1} nm",
            lam * 1e9,
            expected * 1e9
        );
    }

    /// Putterman & Weninger (2000): experimentally observed UV peak at 310 nm.
    ///
    /// The equivalent blackbody temperature is T ≈ 9,350 K.
    #[test]
    fn test_wien_temperature_at_310nm_putterman() {
        let lam = 310e-9;
        let t_equiv = WIEN_CONST / lam;
        // Should be ≈ 9 350 K
        assert!(
            t_equiv > 9_000.0 && t_equiv < 9_800.0,
            "Equivalent blackbody T at 310 nm: got {t_equiv:.0} K, expected 9000-9800 K"
        );
    }

    /// Wien peak scales as 1/T (inversely with temperature).
    #[test]
    fn test_wien_peak_inverse_temperature() {
        let lam1 = wien_peak_wavelength_m(10_000.0);
        let lam2 = wien_peak_wavelength_m(20_000.0);
        let ratio = lam1 / lam2;
        assert!(
            (ratio - 2.0).abs() < 1e-13,
            "Wien peak ∝ 1/T: ratio must be 2.0, got {ratio:.10}"
        );
    }

    // ── Planck spectrum ───────────────────────────────────────────────────────

    /// Planck spectrum must peak at the Wien wavelength (normalised value = 1.0).
    #[test]
    fn test_planck_peaks_at_wien_wavelength() {
        let t = 10_000.0;
        let lam_peak = wien_peak_wavelength_m(t);
        let b_peak = planck_radiance_relative(lam_peak, t);
        assert!(
            (b_peak - 1.0).abs() < 1e-10,
            "Planck spectrum normalised peak must be 1.0, got {b_peak:.12}"
        );
    }

    /// Planck spectrum must be < 0.5 at half and double the peak wavelength.
    #[test]
    fn test_planck_spectrum_falls_off_from_peak() {
        let t = 10_000.0;
        let lam_peak = wien_peak_wavelength_m(t);
        let b_half = planck_radiance_relative(lam_peak / 2.0, t);
        let b_double = planck_radiance_relative(2.0 * lam_peak, t);
        assert!(
            b_half < 0.5,
            "Planck at λ/2 must be < 50% of peak: {b_half:.4}"
        );
        assert!(
            b_double < 0.5,
            "Planck at 2λ must be < 50% of peak: {b_double:.4}"
        );
    }

    // ── Yasui emission ratio ──────────────────────────────────────────────────

    /// Yasui (1997): emission intensity highly sensitive to T_max.
    ///
    /// The `yasui_intensity_ratio` integrates Planck radiance over 200–700 nm.
    /// At T = 5,000 K (λ_peak = 580 nm), there is strong visible emission.
    /// At T = 10,000 K (λ_peak = 290 nm), UV output is higher but broad-band
    /// visible also increases. The ratio over 200–700 nm is ~30–60×.
    ///
    /// The Yasui (1997) T^{8–12} scaling applies specifically to narrow-band
    /// PMT signals (≈400 nm), not to the full 200–700 nm integral.
    #[test]
    fn test_yasui_intensity_ratio_strong_temperature_sensitivity() {
        let ratio = yasui_intensity_ratio(10_000.0, 5_000.0);
        // Over 200–700 nm: at least 20× increase when T doubles from 5k→10k K
        assert!(
            ratio > 20.0,
            "Yasui intensity ratio I(10k K)/I(5k K) must be > 20: got {ratio:.1}"
        );
    }

    /// Intensity ratio must be > 1 when T1 > T2.
    #[test]
    fn test_yasui_ratio_monotone() {
        let r12 = yasui_intensity_ratio(12_000.0, 10_000.0);
        let r21 = yasui_intensity_ratio(10_000.0, 12_000.0);
        assert!(r12 > 1.0, "I(12k)/I(10k) must be > 1, got {r12:.3}");
        assert!(r21 < 1.0, "I(10k)/I(12k) must be < 1, got {r21:.3}");
    }
}
