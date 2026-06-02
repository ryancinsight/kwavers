use kwavers_core::error::KwaversResult;
use kwavers_domain::imaging::photoacoustic::{
    PhotoacousticScenario, PhotoacousticSimulation, PhotoacousticValidationReport,
};
use super::vertical::PhotoacousticPipeline;

/// Thin orchestration layer that owns the canonical photoacoustic pipeline.
#[derive(Debug, Default)]
pub struct PhotoacousticOrchestrator {
    pipeline: PhotoacousticPipeline,
}

impl PhotoacousticOrchestrator {
    /// Execute.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn execute(
        &self,
        scenario: &PhotoacousticScenario,
    ) -> KwaversResult<(PhotoacousticSimulation, PhotoacousticValidationReport)> {
        self.pipeline.simulate(scenario)
    }
}
