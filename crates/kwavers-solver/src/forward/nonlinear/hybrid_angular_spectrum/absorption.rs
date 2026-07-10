//! Absorption operator with power-law attenuation model.
//!
//! ## Algorithm — Exponential Decay (Szabo 1994; Hamilton & Blackstock 2008)
//!
//! For a plane wave propagating distance Δz with power-law attenuation:
//!
//! ```text
//! p(z + Δz) = p(z) · exp(−α(f) · Δz)
//! ```
//!
//! where the frequency-dependent attenuation coefficient is:
//!
//! ```text
//! α(f) = α₀ · (f / f_ref)^y    [Np/m]
//! ```
//!
//! with `α₀` converted from dB/(cm·MHz^y) to Np/m via the exact factor
//! `NP_TO_DB = 20/ln(10)`:
//!
//! ```text
//! α₀ [Np/m] = α₀ [dB/cm] · 100 [cm/m] / NP_TO_DB
//! ```
//!
//! ## References
//!
//! - Szabo TL (1994). "Time domain wave equations for lossy media obeying a
//!   frequency power law." J. Acoust. Soc. Am. 96(1), 491–500.
//!   DOI: 10.1121/1.410434
//! - Hamilton MF, Blackstock DT (2008). *Nonlinear Acoustics*.
//!   Acoustical Society of America Press, §2.3.

use super::HASConfig;
use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_core::constants::numerical::{CM_TO_M, MHZ_TO_HZ};
use kwavers_core::error::KwaversResult;
use leto::{
    Array2,
    Array3,
};

/// Absorption operator implementing power-law frequency-dependent attenuation.
#[derive(Debug)]
pub struct HasAbsorptionOperator {
    /// Attenuation coefficient in dB/(cm·MHz^y) at the reference frequency.
    attenuation_coeff: f64,
    /// Power-law frequency exponent y (dimensionless; typically 1–2 for tissue).
    power_law_exp: f64,
    /// Reference frequency in Hz used for the power-law scaling.
    reference_freq: f64,
}

impl HasAbsorptionOperator {
    /// Construct an `HasAbsorptionOperator` from a validated `HASConfig`.
    ///
    /// The config is re-validated at this boundary ([`HASConfig::validate`]) so a
    /// config reached via `default()` + field mutation — bypassing
    /// [`HASConfig::new`] — cannot drive the power-law absorption with a
    /// non-positive reference frequency or a negative attenuation coefficient.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `config` fails [`HASConfig::validate`].
    pub fn new(config: &HASConfig) -> KwaversResult<Self> {
        config.validate()?;
        Ok(Self {
            attenuation_coeff: config.attenuation_coeff,
            power_law_exp: config.power_law_exponent,
            reference_freq: config.reference_frequency,
        })
    }

    /// Apply CW absorption over propagation step `dz`.
    ///
    /// ## Theorem (Szabo 1994, Eq. 1)
    ///
    /// For monochromatic pressure at the reference frequency `f_ref`:
    ///
    /// ```text
    /// p(z + Δz, t) = p(z, t) · exp(−α(f_ref) · Δz)
    /// ```
    ///
    /// where `α(f_ref) = α₀ · (f_ref/MHz)^y · 100 / NP_TO_DB` [Np/m].
    ///
    /// ## Parameters
    /// - `pressure` — spatial pressure field `p[ix, iy, iz]` (Pa)
    /// - `dz`       — propagation step (m)
    ///
    /// ## References
    /// - Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500.
    ///   DOI: 10.1121/1.410434
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply(&self, pressure: &Array3<f64>, dz: f64) -> KwaversResult<Array3<f64>> {
        let freq_mhz = self.reference_freq / MHZ_TO_HZ;
        // dB/(cm·MHz^y) → Np/m: divide by NP_TO_DB and CM_TO_M (SSOT).
        let alpha_db_cm = self.attenuation_coeff * freq_mhz.powf(self.power_law_exp);
        let alpha_np_m = alpha_db_cm / NP_TO_DB / CM_TO_M;
        let factor = (-alpha_np_m * dz).exp();
        Ok(pressure.mapv(|p| p * factor))
    }

    /// Apply broadband (multi-harmonic) power-law absorption in-place.
    ///
    /// ## Theorem (Hamilton & Blackstock 2008, §2.3)
    ///
    /// For the n-th harmonic (n = 1, 2, …) at frequency `n·f₀`, the
    /// attenuation coefficient scales as:
    ///
    /// ```text
    /// α_n = α₀ · n^y    [Np/m]
    /// ```
    ///
    /// Each harmonic pressure plane decays independently:
    ///
    /// ```text
    /// P_n(z + Δz) = P_n(z) · exp(−α_n · Δz)
    /// ```
    ///
    /// ## Parameters
    /// - `harmonics` — mutable slice of 2-D pressure planes, one per harmonic.
    ///   `harmonics[0]` is the fundamental (n=1),
    ///   `harmonics[1]` is the 2nd harmonic (n=2), etc.
    /// - `dz`        — propagation step (m)
    /// - `f0`        — fundamental frequency (Hz) (used for unit conversion only;
    ///   the power-law exponent already encodes the frequency scaling)
    ///
    /// ## References
    /// - Hamilton MF, Blackstock DT (2008). *Nonlinear Acoustics*. ASA Press, §2.3.
    pub fn apply_broadband(&self, harmonics: &mut [Array2<f64>], dz: f64, f0: f64) {
        let freq_mhz = f0 / MHZ_TO_HZ;
        // α₀ at f₀ in Np/m (SSOT unit conversions).
        let alpha_np_f0 =
            self.attenuation_coeff * freq_mhz.powf(self.power_law_exp) / NP_TO_DB / CM_TO_M;
        for (n, plane) in harmonics.iter_mut().enumerate() {
            let harmonic_order = (n + 1) as f64; // n=0 → 1st harmonic
                                                 // α_n = α₀ · n^y  (power-law frequency scaling)
            let alpha_n = alpha_np_f0 * harmonic_order.powf(self.power_law_exp);
            let decay = (-alpha_n * dz).exp();
            plane.iter_mut().for_each(|v| *v *= decay);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use leto::Array2;

    fn default_config() -> HASConfig {
        HASConfig::default()
    }

    /// Zero attenuation coefficient → field is unchanged (identity).
    ///
    /// Theorem: exp(−0·Δz) = 1 for any Δz.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_absorption_zero_attenuation_identity() {
        let mut config = default_config();
        config.attenuation_coeff = 0.0;
        let op = HasAbsorptionOperator::new(&config).unwrap();

        let pressure = Array3::from_shape_fn((4, 4, 4), |[i, j, k]| (i + j + k) as f64 + 1.0);
        let result = op.apply(&pressure, 0.05).unwrap();

        for ((_, &orig), &out) in pressure.indexed_iter().zip(result.iter()) {
            assert_abs_diff_eq!(out, orig, epsilon = 1e-15);
        }
    }

    /// Known decay: α=10 Np/m, dz=0.01 m → factor = exp(−0.1).
    ///
    /// ## Derivation
    /// α_db_cm = α_coeff · (f_MHz)^y;  need to recover 10 Np/m.
    /// At f=1 MHz, y=1: α_coeff = 10 · NP_TO_DB / 100.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_absorption_known_decay_formula() {
        use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
        let mut config = default_config();
        config.power_law_exponent = 1.0;
        config.reference_frequency = MHZ_TO_HZ; // 1 MHz
                                                // Set so that α_np_m = 10.0 Np/m
        config.attenuation_coeff = 10.0 * NP_TO_DB / 100.0; // dB/cm/MHz

        let op = HasAbsorptionOperator::new(&config).unwrap();
        let dz = 0.01; // 1 cm
        let expected_factor = (-10.0_f64 * dz).exp(); // exp(-0.1)

        let pressure = Array3::from_elem((3, 3, 3), 1.0_f64);
        let result = op.apply(&pressure, dz).unwrap();

        assert_abs_diff_eq!(result[[1, 1, 1]], expected_factor, epsilon = 1e-12);
    }

    /// Broadband: single harmonic (n=1) must equal CW `apply()` at the same frequency.
    ///
    /// ## Theorem
    /// For a single-element `harmonics` slice, `apply_broadband` applies
    /// `α₀ · 1^y = α₀` — identical to `apply()` at the reference frequency.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_absorption_broadband_cw_agreement() {
        let config = default_config();
        let op = HasAbsorptionOperator::new(&config).unwrap();
        let f0 = config.reference_frequency;
        let dz = 0.005;

        // CW path: apply() uses reference_frequency
        let p3d = Array3::from_elem((4, 4, 1), 2.0_f64);
        let cw_result = op.apply(&p3d, dz).unwrap();
        let cw_val = cw_result[[1, 1, 0]];

        // Broadband path with a single fundamental harmonic (n=1)
        let mut harmonics = vec![Array2::from_elem((4, 4), 2.0_f64)];
        op.apply_broadband(&mut harmonics, dz, f0);
        let bb_val = harmonics[0][[1, 1]];

        assert_abs_diff_eq!(bb_val, cw_val, epsilon = 1e-12);
    }

    /// Broadband 2nd harmonic decays faster than fundamental (power-law).
    ///
    /// At y=2: α₂ = α₀·2² = 4·α₀ → steeper decay than α₀ for α₀ > 0, dz > 0.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_broadband_higher_harmonic_decays_faster() {
        let mut config = default_config();
        config.power_law_exponent = 2.0;
        config.attenuation_coeff = 0.5;
        let op = HasAbsorptionOperator::new(&config).unwrap();
        let dz = 0.01;
        let f0 = config.reference_frequency;

        let mut harmonics = vec![
            Array2::from_elem((2, 2), 1.0_f64), // fundamental (n=1)
            Array2::from_elem((2, 2), 1.0_f64), // 2nd harmonic (n=2)
        ];
        op.apply_broadband(&mut harmonics, dz, f0);

        // The 2nd harmonic must decay faster than fundamental (y=2 → α₂ = 4·α₁)
        assert!(
            harmonics[1][[0, 0]] < harmonics[0][[0, 0]],
            "2nd harmonic ({}) should decay faster than fundamental ({})",
            harmonics[1][[0, 0]],
            harmonics[0][[0, 0]]
        );
    }

    /// NP_TO_DB constant is used: result differs measurably from 8.686 approximation.
    ///
    /// Relative difference between exact NP_TO_DB and 8.686 is ~1.3e-4.
    /// Over 1000 PSTD steps this would accumulate to ~0.13 dB error.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_np_to_db_exact_constant_used() {
        use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
        // The exact value and the approximation should differ
        assert!(
            (NP_TO_DB - 8.686_f64).abs() > 1e-5,
            "NP_TO_DB={NP_TO_DB} should differ from 8.686 by ~1.3e-4"
        );
        // NP_TO_DB = 20/ln(10) exactly
        let exact = 20.0 / f64::ln(10.0);
        assert_abs_diff_eq!(NP_TO_DB, exact, epsilon = f64::EPSILON * 10.0);
    }

    /// SOL-5: the operator rejects a config whose validation invariants were
    /// broken by post-`default` field mutation (the bypass around `HASConfig::new`).
    /// A non-positive reference frequency would otherwise drive `(f/MHz)^y → NaN`,
    /// and a negative attenuation coefficient is unphysical.
    #[test]
    fn test_absorption_rejects_invalid_config() {
        let mut bad_freq = default_config();
        bad_freq.reference_frequency = 0.0;
        assert!(
            HasAbsorptionOperator::new(&bad_freq).is_err(),
            "non-positive reference frequency must be rejected"
        );

        let mut neg_atten = default_config();
        neg_atten.attenuation_coeff = -1.0;
        assert!(
            HasAbsorptionOperator::new(&neg_atten).is_err(),
            "negative attenuation coefficient must be rejected"
        );

        let mut bad_exp = default_config();
        bad_exp.power_law_exponent = 4.0; // outside [0, 3]
        assert!(
            HasAbsorptionOperator::new(&bad_exp).is_err(),
            "out-of-range power-law exponent must be rejected"
        );

        // A valid (default) config still constructs.
        assert!(HasAbsorptionOperator::new(&default_config()).is_ok());
    }
}
