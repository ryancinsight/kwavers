//! Canonical photoacoustic solver vertical.
//!
//! The retained public surface is organized by physical stage:
//! optical transport, thermoelastic source generation, acoustic propagation,
//! and reconstruction. Each stage owns its own workspace and validation types
//! so allocation policy and scientific evidence stay local to the algorithm
//! that consumes them.

pub mod acoustic;
pub mod api;
pub mod optical;
mod pipeline;
pub mod reconstruction;
pub mod source;
pub mod validation;

pub use acoustic::{gpu_acoustic_available, AcousticForwardModel, AcousticGpuWorkspace};
pub use optical::{
    DiffusionOpticalSolver, MonteCarloOpticalSolver, OpticalForwardModel, OpticalSolveResult,
};
pub use pipeline::{
    PhotoacousticBenchmarkCase, PhotoacousticPipeline, PhotoacousticValidationCase,
    PhotoacousticWorkspace,
};
pub use reconstruction::PhotoacousticReconstructionModel;
pub use source::PhotoacousticSourceModel;
pub use validation::validate_photoacoustic_simulation;
