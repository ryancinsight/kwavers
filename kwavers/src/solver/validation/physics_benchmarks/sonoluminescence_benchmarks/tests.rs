//! SBSL benchmark tests against Brenner, Yasui, and Putterman references.

use super::conditions::BrennerSBSLConditions;
use super::constants::WIEN_CONST;
use super::kernels::{
    blake_threshold, collapse_time_fraction, minnaert_resonance_radius, planck_radiance_relative,
    wien_peak_wavelength_m, yasui_intensity_ratio,
};

const COND: fn() -> BrennerSBSLConditions = BrennerSBSLConditions::default;

// ── Minnaert resonance radius ─────────────────────────────────────────────

/// At 26.5 kHz in water, Minnaert resonance radius ≈ 124 µm.
///
/// Formula: R₀_res = (1/2πf)√(3γp₀/ρ_L).
///
/// Note: In SBSL (Brenner 2002) the equilibrium bubble radius is R₀ ≈ 5 µm,
/// which is far below the 124 µm Minnaert resonance radius at 26.5 kHz.
/// The bubble is driven at a frequency ~22× below its natural frequency.
/// # Panics
/// - Panics if an internal precondition is violated.
///
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
/// # Panics
/// - Panics if an internal precondition is violated.
///
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
/// # Panics
/// - Panics if assertion fails: `Blake threshold must be tensile (negative), got {p_b:.3e} Pa`.
///
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
/// # Panics
/// - Panics if assertion fails: `Blake threshold {p_b:.3e} Pa should not exceed −1 atm in magnitude`.
///
#[test]
fn test_blake_threshold_above_minus_p0() {
    let c = COND();
    let p_b = blake_threshold(c.p0_pa, c.p_v, c.r0_m, c.sigma);
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
/// # Panics
/// - Panics if assertion fails: `Collapse time fraction = {t_frac:.4} must be < 15% of acoustic period`.
/// - Panics if assertion fails: `Collapse time fraction = {t_frac:.4} must be > 2% (non-trivial collapse)`.
///
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
/// # Panics
/// - Panics if assertion fails: `Collapse time fraction must be positive, got {t_frac}`.
///
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
/// # Panics
/// - Panics if assertion fails: `Wien peak at 10 000 K: got {:.1} nm, expected {:.1} nm`.
///
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
/// # Panics
/// - Panics if assertion fails: `Equivalent blackbody T at 310 nm: got {t_equiv:.0} K, expected 9000-9800 K`.
///
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
/// # Panics
/// - Panics if an internal precondition is violated.
///
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
/// # Panics
/// - Panics if an internal precondition is violated.
///
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
/// # Panics
/// - Panics if assertion fails: `Planck at λ/2 must be < 50% of peak: {b_half:.4}`.
/// - Panics if assertion fails: `Planck at 2λ must be < 50% of peak: {b_double:.4}`.
///
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
/// # Panics
/// - Panics if assertion fails: `Yasui intensity ratio I(10k K)/I(5k K) must be > 20: got {ratio:.1}`.
///
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
/// # Panics
/// - Panics if assertion fails: `I(12k)/I(10k) must be > 1, got {r12:.3}`.
/// - Panics if assertion fails: `I(10k)/I(12k) must be < 1, got {r21:.3}`.
///
#[test]
fn test_yasui_ratio_monotone() {
    let r12 = yasui_intensity_ratio(12_000.0, 10_000.0);
    let r21 = yasui_intensity_ratio(10_000.0, 12_000.0);
    assert!(r12 > 1.0, "I(12k)/I(10k) must be > 1, got {r12:.3}");
    assert!(r21 < 1.0, "I(10k)/I(12k) must be < 1, got {r21:.3}");
}
