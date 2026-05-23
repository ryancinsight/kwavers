//! Imaging resolution and signal quality theorem validators.

use super::super::{TheoremValidation, TheoremValidator};
use crate::core::constants::numerical::MHZ_TO_HZ;

impl TheoremValidator {
    /// Validate MUSIC algorithm resolution theorem (Rayleigh criterion + Cramér-Rao bound)
    #[must_use]
    pub fn validate_music_resolution(
        array_length: f64,
        wavelength: f64,
        snr_db: f64,
        measured_resolution: f64,
    ) -> TheoremValidation {
        let rayleigh_limit = 0.886 * wavelength / array_length;
        let snr_linear = 10.0f64.powf(snr_db / 10.0);
        let cramer_rao_limit = rayleigh_limit / snr_linear.sqrt();

        let passed = measured_resolution >= cramer_rao_limit * 0.9;

        TheoremValidation {
            theorem: "MUSIC Resolution Theorem".to_owned(),
            passed,
            error_bound: cramer_rao_limit,
            measured_error: measured_resolution,
            confidence: if passed { 0.9 } else { 0.3 },
            details: format!(
                "Rayleigh limit: {:.2e} rad, CR bound: {:.2e} rad, Measured: {:.2e} rad",
                rayleigh_limit, cramer_rao_limit, measured_resolution
            ),
        }
    }

    /// Validate ultrasound axial resolution: Δr = c / (2 · bandwidth)
    #[must_use]
    pub fn validate_sa_resolution(
        bandwidth: f64,
        sound_speed: f64,
        _snr_db: f64,
        measured_resolution: f64,
    ) -> TheoremValidation {
        let theoretical_resolution = sound_speed / (2.0 * bandwidth);
        let passed = measured_resolution <= theoretical_resolution * 2.0;

        TheoremValidation {
            theorem: "Ultrasound Axial Resolution".to_owned(),
            passed,
            error_bound: theoretical_resolution,
            measured_error: measured_resolution,
            confidence: if passed { 0.9 } else { 0.5 },
            details: format!(
                "Bandwidth: {:.1} MHz, c: {:.0} m/s, Theoretical: {:.2e} m, Measured: {:.2e} m, Ratio: {:.1}x",
                bandwidth / MHZ_TO_HZ, sound_speed, theoretical_resolution, measured_resolution,
                measured_resolution / theoretical_resolution
            ),
        }
    }

    /// Validate coded excitation SNR: SNR_improved = SNR_original · √L · η
    #[must_use]
    pub fn validate_coded_excitation_snr(
        code_length: usize,
        compression_ratio: f64,
        measured_snr_improvement_db: f64,
    ) -> TheoremValidation {
        let processing_gain_db = 5.0 * (code_length as f64).log10();
        let compression_gain_db = 10.0 * compression_ratio.log10();
        let theoretical_snr_db = processing_gain_db + compression_gain_db;

        let passed = measured_snr_improvement_db >= theoretical_snr_db * 0.7;
        let efficiency = measured_snr_improvement_db / theoretical_snr_db;

        TheoremValidation {
            theorem: "Coded Excitation SNR Theorem".to_owned(),
            passed,
            error_bound: theoretical_snr_db,
            measured_error: measured_snr_improvement_db,
            confidence: if passed { 0.85 } else { 0.4 },
            details: format!(
                "Code length: {}, Processing gain: {:.1} dB, Compression: {:.1}x ({:.1} dB), Theoretical: {:.1} dB, Measured: {:.1} dB, Efficiency: {:.1}%",
                code_length, processing_gain_db, compression_ratio, compression_gain_db,
                theoretical_snr_db, measured_snr_improvement_db, efficiency * 100.0
            ),
        }
    }
}
