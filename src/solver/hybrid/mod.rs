//! Hybrid PSTD/FDTD solver combining the strengths of both methods
//!
//! This module implements an intelligent hybrid solver that adaptively
//! selects between PSTD and FDTD methods based on local field characteristics.
//!
//! ## Overview
//!
//! The hybrid solver leverages the advantages of both Pseudo-Spectral Time Domain (PSTD)
//! and Finite-Difference Time Domain (FDTD) methods:
//!
//! - **PSTD**: Spectral precision, no numerical dispersion, suitable for smooth fields
//! - **FDTD**: Comprehensive shock handling, local operations, suitable for discontinuities
//!
//! ## Architecture
//!
//! The solver uses a domain decomposition approach where different regions of the
//! computational domain can use different numerical methods based on local field
//! characteristics.

pub mod adaptive_selection;
pub mod config;
pub mod coupling; // Domain coupling interface
pub mod domain_decomposition;
pub mod metrics;
pub mod plugin;
pub mod solver;
pub mod validation;

// Re-export main types
pub use config::{
    CouplingInterfaceConfig, DecompositionStrategy, HybridConfig, OptimizationConfig,
    ValidationConfig,
};
pub use metrics::{EfficiencyMetrics, HybridMetrics, ValidationResults};
pub use plugin::HybridPlugin;
pub use solver::HybridSolver;
