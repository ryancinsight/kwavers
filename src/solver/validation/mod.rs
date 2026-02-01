pub mod config;
pub mod numerical_accuracy;
pub mod physics_benchmarks;

// ============================================================================
// EXPLICIT RE-EXPORTS (Validation API)
// ============================================================================

/// Validation configuration
pub use config::ValidationParameters;

/// Analytical solutions for validation
pub use physics_benchmarks::{PlaneWaveSolution, PointSourceSolution, StandingWaveSolution};

/// Benchmark parameters
pub use physics_benchmarks::{GaussianBeamParameters, TissueProperties};
