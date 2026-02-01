pub mod config;
pub mod numerical_accuracy;
pub mod physics_benchmarks;

// ============================================================================
// EXPLICIT RE-EXPORTS (Validation API)
// ============================================================================

/// Validation configuration
pub use config::ValidationParameters;

/// Benchmark parameters
pub use physics_benchmarks::GaussianBeamParameters;

// Module planned but not yet implemented:
// - PlaneWaveSolution: Analytical solution for plane wave propagation
// - PointSourceSolution: Analytical solution for point source radiation
// - StandingWaveSolution: Analytical solution for standing wave patterns
// - TissueProperties: Tissue parameter definitions for benchmarking
