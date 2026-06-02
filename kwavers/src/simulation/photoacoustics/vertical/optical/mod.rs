use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{OpticalModel, PhotoacousticScenario};
use ndarray::Array3;

mod benchmarks;
mod diffusion;
mod monte_carlo;
mod validation;
mod workspace;

pub use benchmarks::OpticalBenchmarkCase;
pub use diffusion::DiffusionOpticalSolver;
pub use monte_carlo::MonteCarloOpticalSolver;
pub use validation::{validate_diffusion_regime, OpticalValidationCase};
pub use workspace::OpticalWorkspace;

/// Output of a canonical optical forward solve.
#[derive(Debug, Clone)]
pub struct OpticalSolveResult {
    pub model: OpticalModel,
    pub fluence: Array3<f64>,
}

/// Canonical optical forward-model contract.
pub trait OpticalForwardModel: std::fmt::Debug {
    /// Return the optical model variant this implementation provides.
    fn model_kind(&self) -> OpticalModel;

    /// Run the optical forward model for `scenario`.
    ///
    /// # Errors
    /// - Returns [`Err`] if the optical solver fails to converge or encounters invalid input.
    fn solve(&self, scenario: &PhotoacousticScenario) -> KwaversResult<OpticalSolveResult>;
}
