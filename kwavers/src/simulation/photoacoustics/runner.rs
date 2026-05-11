use super::PhotoacousticOrchestrator;
use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{
    PhotoacousticScenario, PhotoacousticSimulation, PhotoacousticValidationReport,
};

#[derive(Debug, Default)]
pub struct PhotoacousticRunner {
    orchestrator: PhotoacousticOrchestrator,
}

impl PhotoacousticRunner {
    /// Run.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn run(
        &self,
        scenario: &PhotoacousticScenario,
    ) -> KwaversResult<(PhotoacousticSimulation, PhotoacousticValidationReport)> {
        self.orchestrator.execute(scenario)
    }
}
