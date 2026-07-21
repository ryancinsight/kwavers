//! Deterministic const-generic correlation screening.

use super::config::SensitivityConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use tyche_core::{CorrelationScreening, Design, LatinHypercube, ParameterSpace, SensitivityReport};

/// Sensitivity analyzer backed by Tyche sampling and statistics.
#[derive(Debug)]
pub struct SensitivityAnalyzer {
    pub(super) config: SensitivityConfig,
}

impl SensitivityAnalyzer {
    /// Create a sensitivity analyzer.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when fewer than two samples are
    /// configured because correlation is then undefined.
    pub fn new(config: SensitivityConfig) -> KwaversResult<Self> {
        if config.sample_count.get() < 2 {
            return Err(KwaversError::InvalidInput(
                "Correlation screening requires at least two samples".to_owned(),
            ));
        }
        Ok(Self { config })
    }

    /// Screen fixed-dimensional parameters by squared Pearson correlation.
    ///
    /// Sampling is random-access and deterministic for the configured seed.
    /// The model returns one scalar response because a multi-output statistic
    /// requires an explicit reduction contract.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] for a non-finite model response
    /// or an invalid provider operation.
    pub fn analyze<const PARAMETERS: usize>(
        &self,
        model: impl Fn(&[f64; PARAMETERS]) -> f64,
        space: &ParameterSpace<'_, f64, PARAMETERS>,
    ) -> KwaversResult<SensitivityReport<f64, PARAMETERS>> {
        let design = LatinHypercube::new(self.config.seed, self.config.sample_count);
        let mut screening = CorrelationScreening::new();
        let mut unit = [0.0; PARAMETERS];
        let mut parameters = [0.0; PARAMETERS];

        for sample in 0..design.sample_count() {
            design
                .sample_unit_into(sample, &mut unit)
                .map_err(|error| {
                    KwaversError::InvalidInput(format!("Tyche sample generation failed: {error}"))
                })?;
            space.map_unit_into(&unit, &mut parameters);
            let response = model(&parameters);
            if !response.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "Sensitivity model returned non-finite response at sample {sample}: {response}"
                )));
            }
            screening.update(&parameters, response);
        }

        screening.report().map_err(|error| {
            KwaversError::InvalidInput(format!("Tyche sensitivity report failed: {error}"))
        })
    }
}
