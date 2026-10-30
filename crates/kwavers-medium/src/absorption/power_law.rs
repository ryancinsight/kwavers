//! Power-law absorption model

use kwavers_core::constants::acoustic_parameters::{ABSORPTION_TISSUE, WATER_ABSORPTION_ALPHA_0};
use kwavers_core::constants::numerical::{CM_TO_M, MHZ_TO_HZ, TWO_PI};
use kwavers_core::constants::{DB_TO_NP, REFERENCE_FREQUENCY_HZ};
use serde::{Deserialize, Serialize};

/// Power-law absorption model configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PowerLawAbsorption {
    /// Absorption coefficient at 1 `MHz` [dB/(MHz^y cm)]
    pub alpha_0: f64,
    /// Power law exponent (typically 1.0-1.5)
    pub y: f64,
    /// Reference frequency for `alpha_0` \[Hz\]
    pub f_ref: f64,
    /// Enable dispersion correction
    pub dispersion_correction: bool,
}

impl Default for PowerLawAbsorption {
    fn default() -> Self {
        Self {
            alpha_0: ABSORPTION_TISSUE, // 0.75 dB/(MHz^y·cm) — Duck (1990) Table 4.1
            y: 1.1,                     // Typical soft tissue exponent
            f_ref: REFERENCE_FREQUENCY_HZ,
            dispersion_correction: true,
        }
    }
}

impl PowerLawAbsorption {
    /// Create absorption model for water at 20°C
    #[must_use]
    pub fn water() -> Self {
        Self {
            alpha_0: WATER_ABSORPTION_ALPHA_0, // 0.0022 dB/(MHz^y·cm) — Duck (1990) Table 4.1
            y: 2.0,                            // Quadratic frequency dependence
            f_ref: REFERENCE_FREQUENCY_HZ,
            dispersion_correction: false,
        }
    }

    /// Create absorption model for soft tissue
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            alpha_0: ABSORPTION_TISSUE,
            y: 1.1,
            f_ref: REFERENCE_FREQUENCY_HZ,
            dispersion_correction: true,
        }
    }

    /// Calculate absorption coefficient at given frequency
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // Convert dB/(MHz^y cm) → Np/m:
        //   α [Np/m] = α [dB/cm] · (1/CM_TO_M) [cm/m] · DB_TO_NP [Np/dB]
        let f_mhz = frequency / MHZ_TO_HZ;
        let alpha_db = self.alpha_0 * f_mhz.powf(self.y);

        alpha_db * DB_TO_NP / CM_TO_M
    }

    /// Phase velocity from the Szabo / Treeby-Cox Kramers-Kronig relation
    /// for power-law absorption.
    ///
    /// ## Formula (Szabo 1994 Eq 13; Treeby & Cox 2010 Eq 11)
    ///
    /// ```text
    /// c_p(ω)⁻¹ − c_0⁻¹ = α(ω) · tan(πy/2) / ω
    /// ```
    ///
    /// where `α(ω) = α(ω)` (Np/m) is obtained from
    /// [`Self::absorption_at_frequency`] (which handles the dB/(cm·MHz^y) →
    /// Np/m conversion).  Inverting:
    ///
    /// ```text
    /// c_p(ω) = c_0 / (1 + c_0 · α(ω) · tan(πy/2) / ω)
    /// ```
    ///
    /// Prior to 2026-05-21 this used `self.alpha_0` directly in the
    /// formula — a value stored in dB/(cm·MHz^y), not the SI Np/m
    /// required by Kramers-Kronig — *and* omitted the `c_0` factor that
    /// scales the inverse-phase-velocity correction.  Combined, the prior
    /// dispersion correction was off by ~10⁵× from the canonical result.
    ///
    /// ## References
    /// - Szabo T. L. (1994). *J. Acoust. Soc. Am.* 96(1), 491–500. Eq 13.
    /// - Treeby B. E. & Cox B. T. (2010). *J. Acoust. Soc. Am.* 127(5), 2741. Eq 11.
    #[must_use]
    pub fn phase_velocity(&self, frequency: f64, c0: f64) -> f64 {
        if !self.dispersion_correction || frequency == 0.0 {
            return c0;
        }

        let omega = TWO_PI * frequency;
        let tan_term = (std::f64::consts::PI * self.y / 2.0).tan();
        let alpha = self.absorption_at_frequency(frequency); // Np/m

        let denom = 1.0 + c0 * alpha * tan_term / omega;
        c0 / denom
    }
}

/// Convert the k-Wave power-law absorption coefficient from
/// `dB/(MHz^y·cm)` to the spectral coefficient `Np/((rad/s)^y·m)`.
///
/// This is the single owner of the rad/s-prefactor conversion
/// (KW-POWER-LAW-INVERSE-FORMS): every fractional-Laplacian path converts
/// through it, never by hand-spelled factors.
///
/// # Theorem
/// For a power-law absorption prefactor `α_dB` defined at `1 MHz`,
/// the Treeby & Cox spectral coefficient `α_SI` satisfies:
///
/// ```text
/// α(f) [dB/cm] = α_dB · (f / 1 MHz)^y
/// α(ω) [Np/m]  = α_SI · ω^y
/// α_SI = α_dB · (ln(10)/20) · 100 · (1 / (2π · 10^6))^y
/// ```
///
/// The last line is the exact `k-Wave` `db2neper` conversion used before the
/// fractional-Laplacian coefficients `τ` and `η` are formed.
#[must_use]
#[inline]
pub fn power_law_db_cm_to_np_omega_m(alpha_db_cm: f64, alpha_power: f64) -> f64 {
    // The `/ CM_TO_M` factor converts dB/cm → dB/m (1/0.01 = 100).
    alpha_db_cm * DB_TO_NP / CM_TO_M * (1.0 / (TWO_PI * REFERENCE_FREQUENCY_HZ)).powf(alpha_power)
}

/// Amplitude attenuation `α(f)` [Np·m⁻¹] at frequency `f` for the k-Wave
/// power-law prefactor `α_dB` [dB/(MHz^y·cm)] and exponent `y`.
///
/// ```text
///   α(f) = α_dB · (ln10/20) · 100 · (f / 1 MHz)^y
/// ```
///
/// This is the single owner of the at-frequency conversion: the conversion a
/// **time-domain** solver needs. The relaxation-spectrum fit
/// (`absorption::relaxation_fit`) is posed in Np·m⁻¹ at a reference
/// frequency, whereas [`power_law_db_cm_to_np_omega_m`] produces the
/// `Np/((rad/s)^y·m)` form that only the spectral fractional-Laplacian path
/// uses. Keeping the forms beside each other is deliberate — the units are
/// the easiest thing in unit conversion to get silently wrong.
///
/// Returns 0 for a non-positive or non-finite frequency.
#[must_use]
#[inline]
pub fn power_law_db_cm_to_np_m(alpha_db_cm: f64, alpha_power: f64, freq_hz: f64) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    alpha_db_cm * (DB_TO_NP / CM_TO_M) * (freq_hz / REFERENCE_FREQUENCY_HZ).powf(alpha_power)
}

/// k-Wave `dB/(MHz^y·cm)` power-law prefactor that reproduces a given
/// amplitude attenuation `α(f)` [Np/m] at frequency `f` for exponent `y`.
///
/// Single owner of the inverse conversion:
///
/// ```text
///   α_db = α_Np_m · (CM_TO_M / DB_TO_NP) · (f_ref / f)^y
/// ```
///
/// Used to inject a non-power-law excess attenuation (e.g. a resonant
/// bubble-cloud α from Commander–Prosperetti) into a power-law medium's
/// prefactor so it produces that attenuation at the drive frequency. Exact at
/// `f`; off-frequency it then scales as the medium's `f^y`.
///
/// Returns 0 for a non-positive or non-finite frequency.
#[must_use]
#[inline]
pub fn np_m_to_power_law_db_cm(alpha_np_m: f64, freq_hz: f64, alpha_power: f64) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    alpha_np_m * (CM_TO_M / DB_TO_NP) * (REFERENCE_FREQUENCY_HZ / freq_hz).powf(alpha_power)
}

/// Exact inverse of [`power_law_db_cm_to_np_omega_m`]: the `dB/(MHz^y·cm)`
/// prefactor for a spectral coefficient `Np/((rad/s)^y·m)`.
///
/// ```text
///   α_dB = α_SI · (2π · 10^6)^y · (cm/m) / (ln10/20)
/// ```
#[must_use]
#[inline]
pub fn np_omega_m_to_power_law_db_cm(alpha_si: f64, alpha_power: f64) -> f64 {
    alpha_si * (TWO_PI * REFERENCE_FREQUENCY_HZ).powf(alpha_power) * CM_TO_M / DB_TO_NP
}

/// `dB/(cm·MHz)` prefactor as `Np/(m·MHz)` at the 1 MHz reference (the y = 1
/// linear case).
///
/// `const`-capable owner for `const` initializers, which cannot call the
/// `powf`-based owners: `α [Np/(m·MHz)] = α_dB · (ln10/20) · 100`.
/// Agrees with [`power_law_db_cm_to_np_m`] at `(y = 1, f = 1 MHz)` to 1 ulp
/// (`·DB·100` vs `·(DB/CM)` round the last bit differently); the DB_TO_NP
/// evaluation is canonical per the k-Wave oracle test below.
#[must_use]
#[inline]
pub const fn db_per_cm_mhz_to_np_per_m_mhz(alpha_db_cm_mhz: f64) -> f64 {
    alpha_db_cm_mhz * DB_TO_NP * 100.0
}

/// k-Wave `dB/(MHz^y·cm)` prefactor as an `Np/(m·Hz^y)` prefactor for solvers
/// that build per-frequency masks as `α(f) = prefac · f_Hz^y` (e.g. KZK).
///
/// Single owner of the Hz^y-prefactor conversion:
///
/// ```text
///   prefac = α_dB · (ln10/20) · 100 / (1 MHz)^y
/// ```
#[must_use]
#[inline]
pub fn power_law_db_cm_to_np_m_hz(alpha_db_cm: f64, alpha_power: f64) -> f64 {
    alpha_db_cm * DB_TO_NP / CM_TO_M / MHZ_TO_HZ.powf(alpha_power)
}

/// Power law model implementation
#[derive(Debug)]
pub struct PowerLawModel {
    config: PowerLawAbsorption,
}

impl PowerLawModel {
    /// Create a new power law model
    #[must_use]
    pub fn new(config: PowerLawAbsorption) -> Self {
        Self { config }
    }

    /// Apply power law absorption in frequency domain
    ///
    /// # Panics
    ///
    /// Panics if the spectrum's frequency axis length differs from
    /// `frequencies.len()`. The frequency response is applied plane-by-plane,
    /// so the two lengths must match.
    pub fn apply_frequency_domain(
        &self,
        spectrum: &mut leto::Array3<eunomia::Complex64>,
        frequencies: &[f64],
        distance: f64,
    ) {
        assert_eq!(
            spectrum.shape()[0],
            frequencies.len(),
            "invariant: power-law absorption frequency count must match spectrum axis 0"
        );

        let [_, ny, nz] = spectrum.shape();
        for (i, &freq) in frequencies.iter().enumerate() {
            if freq > 0.0 {
                let alpha = self.config.absorption_at_frequency(freq);
                let attenuation = (-alpha * distance).exp();
                for j in 0..ny {
                    for k in 0..nz {
                        spectrum[[i, j, k]] *= attenuation;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_relative_eq;
    use eunomia::Complex;
    use leto::Array3;

    #[test]
    fn apply_frequency_domain_scales_frequency_planes() {
        let config = PowerLawAbsorption {
            alpha_0: 1.0,
            y: 1.0,
            f_ref: REFERENCE_FREQUENCY_HZ,
            dispersion_correction: false,
        };
        let model = PowerLawModel::new(config);
        let initial = Complex::new(2.0, -1.0);
        let mut spectrum = Array3::from_elem([2, 2, 1], initial);
        let distance = 0.25;

        model.apply_frequency_domain(&mut spectrum, &[0.0, MHZ_TO_HZ], distance);

        assert_eq!(spectrum[[0, 0, 0]], initial);
        assert_eq!(spectrum[[0, 1, 0]], initial);

        let attenuation = (-config.absorption_at_frequency(MHZ_TO_HZ) * distance).exp();
        let expected = initial * attenuation;
        for j in 0..2 {
            assert_relative_eq!(spectrum[[1, j, 0]].re, expected.re, epsilon = 1e-12);
            assert_relative_eq!(spectrum[[1, j, 0]].im, expected.im, epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "power-law absorption frequency count")]
    fn apply_frequency_domain_rejects_frequency_axis_mismatch() {
        let model = PowerLawModel::new(PowerLawAbsorption::default());
        let mut spectrum = Array3::from_elem([2, 1, 1], Complex::new(1.0, 0.0));

        model.apply_frequency_domain(&mut spectrum, &[MHZ_TO_HZ], 1.0);
    }

    #[test]
    fn np_m_conversion_round_trips_against_its_inverse() {
        for &(alpha_db, y, f) in &[
            (0.5_f64, 1.1_f64, 1.0e6_f64),
            (0.75, 1.5, 3.0e6),
            (0.25, 0.4, 5.0e5),
        ] {
            let np_m = power_law_db_cm_to_np_m(alpha_db, y, f);
            let back = np_m_to_power_law_db_cm(np_m, f, y);
            assert!(
                (back - alpha_db).abs() < 1e-12 * alpha_db,
                "round trip {alpha_db} -> {np_m} -> {back}"
            );
        }
        // At the reference frequency the exponent drops out and the conversion
        // is exactly dB/cm -> Np/m.
        let at_ref = power_law_db_cm_to_np_m(1.0, 1.7, 1.0e6);
        assert!((at_ref - 100.0 / 8.685_889_638_065_035).abs() < 1e-12);
        assert_eq!(power_law_db_cm_to_np_m(1.0, 1.0, 0.0), 0.0);
    }

    #[test]
    fn test_power_law_db_cm_to_np_omega_m_matches_kwave_reference() {
        let got = power_law_db_cm_to_np_omega_m(0.75, 1.5);
        let expected: f64 = 5.482_481_235_081_536e-10;
        // The conversion contains six rounded elementary operations and a
        // libm `powf` call. Sixteen ulps bounds this composed forward error
        // at the reference magnitude while retaining k-Wave as the oracle.
        let tolerance = 16.0 * f64::EPSILON * expected.abs().max(got.abs());
        assert!(
            (got - expected).abs() <= tolerance,
            "conversion mismatch: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }

    #[test]
    fn np_m_to_db_cm_round_trips_at_frequency() {
        // A target attenuation [Np/m] at a drive frequency must convert to a
        // power-law prefactor that reproduces exactly that attenuation at f.
        let alpha_target_np_m = 37.0;
        let f = 0.5e6;
        let y = 1.1;
        let db_cm = np_m_to_power_law_db_cm(alpha_target_np_m, f, y);
        let alpha_si = power_law_db_cm_to_np_omega_m(db_cm, y);
        let omega = 2.0 * std::f64::consts::PI * f;
        let recovered = alpha_si * omega.powf(y);
        assert!(
            (recovered - alpha_target_np_m).abs() <= 1e-9 * alpha_target_np_m,
            "round-trip: target={alpha_target_np_m} Np/m, recovered={recovered} Np/m"
        );
    }

    #[test]
    fn np_m_to_db_cm_guards_nonpositive_frequency() {
        assert_eq!(np_m_to_power_law_db_cm(10.0, 0.0, 1.1), 0.0);
        assert_eq!(np_m_to_power_law_db_cm(10.0, -1.0, 1.1), 0.0);
    }

    #[test]
    fn omega_inverse_round_trips_against_forward() {
        for &(db, y) in &[(0.75_f64, 1.5_f64), (0.5, 1.1), (2.0, 2.0)] {
            let si = power_law_db_cm_to_np_omega_m(db, y);
            let back = np_omega_m_to_power_law_db_cm(si, y);
            assert!(
                (back - db).abs() <= 8.0 * f64::EPSILON * db.abs().max(1e-300),
                "omega round trip {db} -> {si} -> {back}"
            );
        }
    }

    #[test]
    fn hz_prefactor_matches_at_frequency_form_at_1hz() {
        // The Np/(m·Hz^y) prefactor is the at-frequency form at 1 Hz exactly.
        for &(alpha_db, y) in &[(0.5_f64, 1.1_f64), (0.75, 1.5), (2.0, 2.0)] {
            let prefac = power_law_db_cm_to_np_m_hz(alpha_db, y);
            let at_1hz = power_law_db_cm_to_np_m(alpha_db, y, 1.0);
            assert!(
                (prefac - at_1hz).abs() <= 4.0 * f64::EPSILON * at_1hz.abs().max(1e-300),
                "Hz prefactor {prefac} != at-1Hz {at_1hz}"
            );
        }
    }
}
