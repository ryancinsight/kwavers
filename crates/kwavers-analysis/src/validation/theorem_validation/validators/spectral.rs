//! Spectral energy theorem validators (Parseval).

use super::super::{TheoremValidation, TheoremValidator};
use eunomia::Complex64;
use leto::Array1;

impl TheoremValidator {
    /// Validate Parseval's theorem: `∑|x`N`|² = (1/N) ∑|X`K`|²`
    #[must_use]
    pub fn validate_parsevals_theorem(
        time_domain: &Array1<f64>,
        freq_domain: &Array1<Complex64>,
        _sampling_rate: f64,
    ) -> TheoremValidation {
        let n_samples = time_domain.len() as f64;
        let time_energy: f64 = time_domain.iter().map(|&x| x * x).sum::<f64>();
        let freq_energy: f64 = freq_domain
            .iter()
            .map(|&x: &Complex64| x.norm_sqr())
            .sum::<f64>()
            / n_samples;

        let error = (time_energy - freq_energy).abs() / time_energy.abs().max(1e-10);
        let passed = error < 1e-10;

        TheoremValidation {
            theorem: "Parseval's Theorem".to_owned(),
            passed,
            error_bound: 1e-12,
            measured_error: error,
            confidence: Self::calculate_confidence(1e-12, error, passed),
            details: format!(
                "N: {}, Time energy: {:.6e}, Freq energy: {:.6e}, Relative error: {:.2e}",
                n_samples as usize, time_energy, freq_energy, error
            ),
        }
    }
}
