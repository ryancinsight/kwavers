use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{
    PhotoacousticScenario, PhotoacousticSimulation, PhotoacousticValidationReport,
};
use crate::solver::photoacoustics::PhotoacousticPipeline;

/// Thin orchestration layer that owns the canonical photoacoustic pipeline.
#[derive(Debug, Default)]
pub struct PhotoacousticOrchestrator {
    pipeline: PhotoacousticPipeline,
}

impl PhotoacousticOrchestrator {
    pub fn execute(
        &self,
        scenario: &PhotoacousticScenario,
    ) -> KwaversResult<(PhotoacousticSimulation, PhotoacousticValidationReport)> {
        self.pipeline.simulate(scenario)
    }
}
