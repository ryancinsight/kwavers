// Public modules
pub mod config;
pub mod data;
pub mod dg; // Discontinuous Galerkin method
pub mod numerics;
pub mod physics;
pub mod plugin;
pub mod propagator;
pub mod solver;
pub mod utils;

// Re-exports
pub use config::PSTDConfig;
pub use plugin::PSTDPlugin;
pub use solver::PSTDSolver;

// Legacy re-exports/Compatibility
// pub use physics::absorption at root if needed? No, use physics::absorption
// pub use numerics::spectral_correction::SpectralCorrectionConfig at root?
pub use config::KSpaceMethod;
pub use numerics::SpectralCorrectionConfig;

// Re-export shared source types as PSTD types for backward compatibility
pub use crate::domain::source::{GridSource as PSTDSource, SourceMode};
use crate::solver::fdtd::SourceHandler;

// pub mod solver; // Core solver implementation - already declared above

pub type KSpaceConfig = PSTDConfig;
pub type KSpaceSolver = PSTDSolver;
pub type KSpaceSource = PSTDSource;
pub type KSpaceSourceHandler = SourceHandler;
pub type KSpaceSourceMode = crate::domain::source::SourceMode;
