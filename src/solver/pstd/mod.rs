//! Pseudo-Spectral Time Domain (PSTD) solver
//!
//! High-accuracy spectral method for acoustic wave propagation

pub mod config;
pub mod plugin;
pub mod solver;
pub mod spectral_ops;

pub use config::{CorrectionMethod, PstdConfig};
pub use plugin::PstdPlugin;
pub use solver::PstdSolver;
pub use spectral_ops::SpectralOperations;

// Re-export main solver as default
pub use solver::PstdSolver as Pstd;
