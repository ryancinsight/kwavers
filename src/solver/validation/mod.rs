pub mod config;
pub mod numerical_accuracy;
pub mod physics_benchmarks;

// ============================================================================
// EXPLICIT RE-EXPORTS (Validation API)
// ============================================================================

/// Validation configuration
pub use config::ValidationParameters;

/// Analytical solutions for validation
// TODO: Implement PlaneWaveSolution, PointSourceSolution, StandingWaveSolution
// pub use physics_benchmarks::{PlaneWaveSolution, PointSourceSolution, StandingWaveSolution};

/// Benchmark parameters
pub use physics_benchmarks::GaussianBeamParameters;
// TODO: Implement TissueProperties
// pub use physics_benchmarks::TissueProperties;
