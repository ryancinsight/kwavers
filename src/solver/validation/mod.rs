pub mod config;
pub mod kwave_comparison;
pub mod numerical_accuracy;
pub mod physics_benchmarks;

// ============================================================================
// EXPLICIT RE-EXPORTS (Validation API)
// ============================================================================

/// Validation configuration
pub use config::ValidationParameters;

/// Benchmark parameters and utilities
pub use physics_benchmarks::{measure_beam_radius, GaussianBeamParameters};

/// k-Wave comparison and analytical solutions
pub use kwave_comparison::{ErrorMetrics, GaussianBeam, PlaneWave, SphericalWave};

// Module planned but not yet implemented:
// - StandingWaveSolution: Analytical solution for standing wave patterns
// - TissueProperties: Tissue parameter definitions for benchmarking
