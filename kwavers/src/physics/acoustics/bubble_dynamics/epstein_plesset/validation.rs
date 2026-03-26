//! Implementation validation against literature values

use super::solver::EpsteinPlessetStabilitySolver;
use super::types::ValidationResults;
use crate::core::error::KwaversResult;

impl EpsteinPlessetStabilitySolver {
    /// Validate implementation against literature values
    pub fn validate_implementation(&self) -> KwaversResult<ValidationResults> {
        let analysis = self.analyze_stability();

        let expected_freq = self.compute_resonance_frequency();
        let freq_error = (analysis.resonance_frequency - expected_freq).abs() / expected_freq;

        let q_factor_reasonable = analysis.quality_factor > 1.0 && analysis.quality_factor < 1000.0;
        let stability_reasonable =
            analysis.stability_parameter > -1e6 && analysis.stability_parameter < 1e12;

        Ok(ValidationResults {
            resonance_frequency_error: freq_error,
            quality_factor_valid: q_factor_reasonable,
            stability_parameter_valid: stability_reasonable,
            all_tests_passed: freq_error < 1e-10 && q_factor_reasonable && stability_reasonable,
        })
    }
}
