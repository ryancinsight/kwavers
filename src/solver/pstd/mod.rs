//! Pseudo-Spectral Time Domain (PSTD) solver
//! 
//! High-accuracy spectral method for acoustic wave propagation

pub mod config;
pub mod spectral_ops;
pub mod solver;

pub use config::{PstdConfig, CorrectionMethod};
pub use spectral_ops::SpectralOperations;
pub use solver::PstdSolver;

// Re-export main solver as default
pub use solver::PstdSolver as Pstd;